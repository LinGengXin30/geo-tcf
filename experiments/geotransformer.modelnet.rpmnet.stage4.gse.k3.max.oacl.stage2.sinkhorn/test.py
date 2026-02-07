import os.path as osp
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.common import get_log_string, ensure_dir
from geotransformer.utils.torch import release_cuda

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()
        
        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        message = get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message
        
    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']

        # Save txt for TCF
        ref_corr_points = release_cuda(output_dict['ref_corr_points'])
        src_corr_points = release_cuda(output_dict['src_corr_points'])
        ref_corr_log_var = release_cuda(output_dict['ref_corr_log_var'])
        src_corr_log_var = release_cuda(output_dict['src_corr_log_var'])

        # Convert log var to sigma
        ref_sigma = np.exp(0.5 * ref_corr_log_var)
        src_sigma = np.exp(0.5 * src_corr_log_var)

        # Stack: src_x, src_y, src_z, tgt_x, tgt_y, tgt_z, sigma_src, sigma_tgt
        data = np.concatenate([
            src_corr_points,
            ref_corr_points,
            src_sigma[:, None],
            ref_sigma[:, None]
        ], axis=1)

        txt_file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.txt')
        np.savetxt(txt_file_name, data, fmt='%.6f')


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
