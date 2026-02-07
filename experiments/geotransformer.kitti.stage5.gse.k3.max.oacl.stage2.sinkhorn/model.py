import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.modules.geotransformer.rigid_transformation import apply_transform

from backbone import KPConvFPN


class RobustLocalGlobalRegistration(LocalGlobalRegistration):
    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat, ref_knn_log_var, src_knn_log_var):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        global_ref_corr_log_var = ref_knn_log_var[batch_indices, ref_indices]
        global_src_corr_log_var = src_knn_log_var[batch_indices, src_indices]

        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform, global_ref_corr_log_var, global_src_corr_log_var

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
        ref_knn_log_var,
        src_knn_log_var,
    ):
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores, estimated_transform, ref_corr_log_var, src_corr_log_var = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat, ref_knn_log_var, src_knn_log_var
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform, ref_corr_log_var, src_corr_log_var


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = RobustLocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(cfg.backbone.output_dim, cfg.backbone.output_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.backbone.output_dim // 2, 1),
        )

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 7.3 Compute Uncertainty
        ref_node_corr_knn_log_var = self.uncertainty_head(ref_node_corr_knn_feats.view(-1, ref_node_corr_knn_feats.shape[-1]))
        ref_node_corr_knn_log_var = ref_node_corr_knn_log_var.view(ref_node_corr_knn_feats.shape[0], ref_node_corr_knn_feats.shape[1]).squeeze(-1) # (P, K)
        
        src_node_corr_knn_log_var = self.uncertainty_head(src_node_corr_knn_feats.view(-1, src_node_corr_knn_feats.shape[-1]))
        src_node_corr_knn_log_var = src_node_corr_knn_log_var.view(src_node_corr_knn_feats.shape[0], src_node_corr_knn_feats.shape[1]).squeeze(-1) # (P, K)

        output_dict['ref_node_corr_knn_log_var'] = ref_node_corr_knn_log_var
        output_dict['src_node_corr_knn_log_var'] = src_node_corr_knn_log_var

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform, ref_corr_log_var, src_corr_log_var = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
                ref_node_corr_knn_log_var,
                src_node_corr_knn_log_var,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform
            output_dict['ref_corr_log_var'] = ref_corr_log_var
            output_dict['src_corr_log_var'] = src_corr_log_var

        return output_dict


def create_model(cfg):
    model = GeoTransformer(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
