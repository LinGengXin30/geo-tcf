#include "registration/tcf.h"
#include "utils/for_cloud.h"
#include "utils/for_io.h"
#include "utils/for_time.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./demo matches.txt [gt.txt] [resolution]\n";
        return 1;
    }

    std::string path_matches = argv[1];
    std::string path_gt = "";
    float th = 0.30f; // Default resolution for Kitti

    if (argc >= 3) {
        path_gt = argv[2];
    }
    if (argc >= 4) {
        th = std::atof(argv[3]);
    }

    // Load ground-truth pose
    Eigen::Matrix4f gt = Eigen::Matrix4f::Identity(); 
    bool has_gt = false;
    if (!path_gt.empty()) {
        Eigen::MatrixXf gt_dyn;
        loadMatrixDynamic(path_gt, gt_dyn);
        if (gt_dyn.rows() == 4 && gt_dyn.cols() == 4) {
             gt = gt_dyn;
             has_gt = true;
        } else {
             std::cout << "Warning: GT file found but not 4x4.\n";
        }
    }

    std::cout << "Resolution: " << th << " m\n";

    // Load correspondences
    Eigen::MatrixXf matches;
    loadMatrixDynamic(path_matches, matches);
    MatfD3 source_match = matches.leftCols(3);
    MatfD3 target_match = matches.block(0, 3, matches.rows(), 3);

    Matf1D sigma_src = Matf1D::Zero(1, matches.rows());
    Matf1D sigma_tgt = Matf1D::Zero(1, matches.rows());

    std::cout << "Loaded matches with shape: " << matches.rows() << " x " << matches.cols() << "\n";

    if (matches.cols() >= 8) {
        sigma_src = matches.col(6).transpose();
        sigma_tgt = matches.col(7).transpose();
    } else {
        std::cout << "Warning: Uncertainty not found. Using default 0.\n";
    }

    // Start registration
    TicToc tic_tcf;
    std::srand(unsigned(std::time(nullptr)));
    Eigen::Matrix4f trans = twoStageConsensusFilter(source_match, target_match, sigma_src, sigma_tgt, 3*th);
    double time_registration = tic_tcf.toc();
    std::cout << "Runtime: " << time_registration << " ms.\n";

    double re = -1.0;
    double te = -1.0;

    // compute error
    if (has_gt) {
        std::pair<double, double> error = computeTransError(trans, gt);
        re = error.first;
        te = error.second;
        std::cout << "RE: " << re << " deg, TE: " << te << " m.\n";
    } else {
        std::cout << "GT not provided, skipping error computation.\n";
    }

    // Output CSV format for batch evaluation
    std::cout << "CSV_RESULT," << re << "," << te << "," << time_registration << "\n";
    
    return 0;
}