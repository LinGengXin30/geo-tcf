#ifndef TCF_REGISTRATION_TCF_H_
#define TCF_REGISTRATION_TCF_H_
#include "registration/ransac_1pt2pt3pt.h"
#include "registration/irls_welsch.h"

Eigen::Matrix4f twoStageConsensusFilter(MatfD3 match_1, MatfD3 match_2, Eigen::Matrix<float, 2, Eigen::Dynamic> sigmas, float t, float gamma = 2.0, float max_t = -1.0);

#endif
