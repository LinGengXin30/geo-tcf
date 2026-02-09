#include "registration/tcf.h"
#include "utils/for_cloud.h"
#include "utils/for_io.h"
#include "utils/for_time.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

void process_single_pair(const std::string& path_matches, const std::string& path_gt, float th) {
    // Load correspondences
    Eigen::MatrixXf matches;
    loadMatrixDynamic(path_matches, matches);
    MatfD3 source_match = matches.leftCols(3);
    MatfD3 target_match = matches.block(0, 3, matches.rows(), 3);

    Matf1D sigma_src = Matf1D::Zero(1, matches.rows());
    Matf1D sigma_tgt = Matf1D::Zero(1, matches.rows());

    if (matches.cols() >= 8) {
        sigma_src = matches.col(6).transpose();
        sigma_tgt = matches.col(7).transpose();
    }

    // Start registration
    TicToc tic_tcf;
    Eigen::Matrix4f trans = twoStageConsensusFilter(source_match, target_match, sigma_src, sigma_tgt, 3*th);
    double time_registration = tic_tcf.toc();

    double re = -1.0;
    double te = -1.0;

    // Load ground-truth pose
    Eigen::Matrix4f gt = Eigen::Matrix4f::Identity(); 
    bool has_gt = false;
    if (!path_gt.empty()) {
        Eigen::MatrixXf gt_dyn;
        loadMatrixDynamic(path_gt, gt_dyn);
        if (gt_dyn.rows() == 4 && gt_dyn.cols() == 4) {
             gt = gt_dyn;
             has_gt = true;
        }
    }

    // compute error
    if (has_gt) {
        std::pair<double, double> error = computeTransError(trans, gt);
        re = error.first;
        te = error.second;
    }

    // Output CSV format for batch evaluation
    std::cout << "CSV_RESULT," << re << "," << te << "," << time_registration << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: \n";
        std::cout << "  Single mode: ./demo matches.txt [gt.txt] [resolution]\n";
        std::cout << "  Batch mode:  ./demo --batch file_list.txt [resolution]\n";
        return 1;
    }

    std::srand(unsigned(std::time(nullptr)));

    std::string arg1 = argv[1];
    float th = 0.30f; // Default resolution for Kitti

    if (arg1 == "--batch") {
        if (argc < 3) {
            std::cout << "Error: file_list.txt not provided for batch mode.\n";
            return 1;
        }
        std::string list_path = argv[2];
        if (argc >= 4) {
            th = std::atof(argv[3]);
        }

        std::ifstream list_file(list_path);
        if (!list_file.is_open()) {
            std::cout << "Error: Cannot open file list: " << list_path << "\n";
            return 1;
        }

        std::string line;
        while (std::getline(list_file, line)) {
            if (line.empty()) continue;
            // Assuming file list format: "matches_path gt_path" (space separated)
            // or just "matches_path" if no GT
            std::stringstream ss(line);
            std::string matches_path, gt_path;
            ss >> matches_path;
            if (ss >> gt_path) {
                // gt_path read successfully
            } else {
                gt_path = "";
            }
            process_single_pair(matches_path, gt_path, th);
        }
    } else {
        // Single mode
        std::string path_matches = argv[1];
        std::string path_gt = "";
        if (argc >= 3) {
            path_gt = argv[2];
        }
        if (argc >= 4) {
            th = std::atof(argv[3]);
        }
        process_single_pair(path_matches, path_gt, th);
    }
    
    return 0;
}