// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Registration.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <Eigen/Dense>
#include <Core/Utility/Console.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/KDTreeFlann.h>
#include <Core/Registration/Feature.h>

namespace open3d {

namespace {

double compute_distance_squared(const Eigen::Vector3d p1, const Eigen::Vector3d p2){
    Eigen::Vector3d diff = p1 - p2;
    return diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
}

int compute_closest_vertex(const Eigen::Vector3d best_intpoint, int f, const TriangleMesh target){
    int vertex0 = target.triangles_[f][0];
    int vertex1 = target.triangles_[f][1];
    int vertex2 = target.triangles_[f][2];
    double d0 = compute_distance_squared(best_intpoint, target.vertices_[vertex0]);
    double d1 = compute_distance_squared(best_intpoint, target.vertices_[vertex1]); 
    double d2 = compute_distance_squared(best_intpoint, target.vertices_[vertex2]);
    //PrintInfo("d0: %.2f\n", std::sqrt(d0));
    //PrintInfo("d1: %.2f\n", std::sqrt(d1));
    //PrintInfo("d2: %.2f\n", std::sqrt(d2));
    if(d0 <= d1 && d0 <= d2){
        //PrintInfo("picking d0\n");
        return vertex0;
    }
    else{
        if(d1 <= d0 && d1 <= d2){
            //PrintInfo("picking d1\n");
            return vertex1;
        }
        else{
            if(d2 <= d0 && d2 <= d1){
                //PrintInfo("picking d2\n");
                return vertex2;
            }
            else{
                //PrintInfo("Big mistake\n");
                return -1;
            }
        }
    }
}

// Returns true if a ray intersects a triangle. 
bool RayIntersectsTriangle(const Eigen::Vector3d rayOrigin, 
                        const Eigen::Vector3d rayVector, 
                        const Eigen::Vector3d vertex0, 
                        const Eigen::Vector3d vertex1, 
                        const Eigen::Vector3d vertex2,
                        Eigen::Vector3d& outIntersectionPoint)
{
    const float EPSILON = 0.0000001;
    Eigen::Vector3d edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = rayVector.cross(edge2);
    a = edge1.dot(h);
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0/a;
    s = rayOrigin - vertex0;
    u = f * (s.dot(h));
    if (u < 0.0 || u > 1.0)
        return false;
    q = s.cross(edge1);
    v = f * rayVector.dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * edge2.dot(q);
    if (t > EPSILON) // ray intersection
    {
        outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

// For each of the neighboring triangles to each point, find which one is intersected by the line.
void find_sc_corr(const TriangleMesh &target, const std::vector<int> &neigh_triangles,\
        const Eigen::Vector3d &source_point, const Eigen::Vector3d &source_normal,\
        double max_correspondence_distance, \
        std::vector<int> &target_corr, std::vector<double> & sc){

    // For each triangle in the set, find if it is intersecting.
    Eigen::Vector3d best_intpoint(0.0, 0.0, 0.0); 
    double best_intpoint_dist = 100000000;
    int intpoint_face_ix = -1;
    for (int i = 0; i < (int) neigh_triangles.size(); i++){
        Eigen::Vector3i f = target.triangles_[neigh_triangles[i]];

        Eigen::Vector3d intPoint;
        bool intersect = RayIntersectsTriangle(source_point, source_normal, \
                                                target.vertices_[f[0]], target.vertices_[f[1]],\
                                                target.vertices_[f[2]], intPoint);
        if(!intersect){
            continue;
        }

        double dist2 = compute_distance_squared(intPoint, source_point);
        if(dist2 < best_intpoint_dist){
            best_intpoint_dist = dist2;
            best_intpoint = intPoint;
            intpoint_face_ix = neigh_triangles[i];
        }
    }
    if(intpoint_face_ix >= 0){
        int f = intpoint_face_ix;

        // Pick the vertex closest to the intersection point.
        target_corr[0] = compute_closest_vertex(best_intpoint, f, target);
        double dist2 = compute_distance_squared(target.vertices_[target_corr[0]], best_intpoint);
        Eigen::Vector3d v = best_intpoint;
        //PrintInfo("--------------\n");
        //PrintInfo("Intpoint: %.2f %.2f %.2f\n", v[0], v[1], v[2]);
        v = target.vertices_[target_corr[0]];
        //PrintInfo("vrtpoint: %.2f %.2f %.2f\n", v[0], v[1], v[2]);
        //PrintInfo("vrtpoint to intpoint: %.2f\n", std::sqrt(dist2));
        v = source_point;
        //PrintInfo("Srcpoint: %.2f %.2f %.2f\n", v[0], v[1], v[2]);
        dist2 = compute_distance_squared(target.vertices_[target_corr[0]], source_point);

        // Compute shape complementarity
        sc[0] = target.vertex_normals_[target_corr[0]].dot(-source_normal);
        //PrintInfo("normal mult: %f\n", sc[0]);
        sc[0] = sc[0]*std::exp(-0.5*(dist2));
        //PrintInfo("shape comp: %f \n", sc[0]);
    }
    else{
        target_corr[0] = -1;
        sc[0] = 0.0;
    }
}

    //
// PGC 2019: Get registration results based on shape complementarity
RegistrationResult GetRegistrationResultAndCorrespondencesShapeComplementarity(
        const PointCloud &source,
        const TriangleMesh &target,
        const KDTreeFlann &target_faces_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation, 
        const Feature &source_feature,
        const Feature &target_feature,
        const int fitness_type
        ) {

    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return std::move(result);
    }

    double error2 = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        // Compute the correspondences based on shape complementarity. 
        // First find all target points within X of each source point. X=max_correspondence_distance
        // Then get all the triangles on which each target point participates. 
        // Find the intersection of the line between source point/normal  and each of the triangles. 
        // If the intersection is less than 2, we are good.
        // Find the nearest neighbor to the intersection point.
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(20);
            std::vector<double> dists(20);
            std::vector<int> target_corr(1);
            std::vector<double> target_sc(1);
            const auto &point = source.points_[i];
            const auto &normal= source.normals_[i];
            // Identify all faces within X of the target, up to 20
            if (target_faces_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           20, indices, dists) > 0) {
                // Find the intersection of the line between source point/normal 
                //      and each of the triangles.
                find_sc_corr(target, indices, point, normal, max_correspondence_distance, \
                        target_corr, target_sc);

                if(target_sc[0] > 0){
                    //PrintInfo("Adding correspondences: %d %d\n", i, target_corr[0]);
                    error2_private += target_sc[0];
                    correspondence_set_private.push_back(
                        Eigen::Vector2i(i, target_corr[0]));
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
#ifdef _OPENMP
    }
#endif

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 1000.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        //result.fitness_ = (double)corres_number / (double)source.points_.size();
        if(fitness_type == 1){
            // PGC 2019: fitness: the number of correspondences
            result.fitness_ = (double)corres_number; 
        }
        else{
            if(fitness_type == 2){
                // Compute the descriptor distances for all pairs of correspondences.
                double myfitness = 0;
                for (int i = 0; i < (int)result.correspondence_set_.size(); i++) {
                    int s_vix =  result.correspondence_set_[i][0];
                    int t_vix = result.correspondence_set_[i][1];
                    Eigen::VectorXd feat_s = Eigen::VectorXd(source_feature.data_.col(s_vix));
                    Eigen::VectorXd feat_t = Eigen::VectorXd(target_feature.data_.col(t_vix));
                    double desc_dist = 0.0;
//                    std::cout << "Dimension = " << source_feature.Dimension() << std::endl;
//                    std::cout << "Size = " << source_feature.Num() << std::endl;
                    for (int j = 0; j < source_feature.Dimension(); j++){
                        double dist = feat_t[j] - feat_s[j];
                        dist = dist*dist; 
                        desc_dist += dist;
                    }

                    myfitness += 1.0/desc_dist;
                }
                 
                result.fitness_ = myfitness;
            
            }
            else{
                if (fitness_type == 3.0){ // Use shape complementarity (which is in the error).
                    result.fitness_ = error2 / (double)source.points_.size();
                }
                else{
                    result.fitness_ = (double)corres_number / (double)source.points_.size();
                }
            }
        }
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return std::move(result);
}

// PGC 2019: include the fitness type and the features for a custom fitness function.
RegistrationResult GetRegistrationResultAndCorrespondencesCustom(
        const PointCloud &source,
        const PointCloud &target,
        const KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation, 
        const Feature &source_feature,
        const Feature &target_feature,
        const int fitness_type
        ) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return std::move(result);
    }

    double error2 = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           1, indices, dists) > 0) {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                        Eigen::Vector2i(i, indices[0]));
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
#ifdef _OPENMP
    }
#endif

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        //result.fitness_ = (double)corres_number / (double)source.points_.size();
        if(fitness_type == 1){
            // PGC 2019: fitness: the number of correspondences
            result.fitness_ = (double)corres_number; 
        }
        else{
            if(fitness_type == 2){
                // Compute the descriptor distances for all pairs of correspondences.
                double myfitness = 0;
                for (int i = 0; i < (int)result.correspondence_set_.size(); i++) {
                    int s_vix =  result.correspondence_set_[i][0];
                    int t_vix = result.correspondence_set_[i][1];
                    Eigen::VectorXd feat_s = Eigen::VectorXd(source_feature.data_.col(s_vix));
                    Eigen::VectorXd feat_t = Eigen::VectorXd(target_feature.data_.col(t_vix));
                    double desc_dist = 0.0;
//                    std::cout << "Dimension = " << source_feature.Dimension() << std::endl;
//                    std::cout << "Size = " << source_feature.Num() << std::endl;
                    for (int j = 0; j < source_feature.Dimension(); j++){
                        double dist = feat_t[j] - feat_s[j];
                        dist = dist*dist; 
                        desc_dist += dist;
                    }

                    myfitness += 1.0/desc_dist;
                }
                 
                result.fitness_ = myfitness;
            
            }
            else{
                result.fitness_ = (double)corres_number / (double)source.points_.size();
            }
        }
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return std::move(result);
}

RegistrationResult GetRegistrationResultAndCorrespondences(
        const PointCloud &source,
        const PointCloud &target,
        const KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return std::move(result);
    }

    double error2 = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           1, indices, dists) > 0) {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                        Eigen::Vector2i(i, indices[0]));
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
#ifdef _OPENMP
    }
#endif

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        //result.fitness_ = (double)corres_number / (double)source.points_.size();
        // PGC 2019: fitness is the number of correspondences, not the fraction.
        result.fitness_ = (double)corres_number; 
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return std::move(result);
}

RegistrationResult EvaluateRANSACBasedOnCorrespondence(
        const PointCloud &source,
        const PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);
    double error2 = 0.0;
    int good = 0;
    double max_dis2 = max_correspondence_distance * max_correspondence_distance;
    for (const auto &c : corres) {
        double dis2 =
                (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
        if (dis2 < max_dis2) {
            good++;
            error2 += dis2;
        }
    }
    if (good == 0) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        result.fitness_ = (double)good / (double)corres.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)good);
    }
    return result;
}

}  // unnamed namespace

RegistrationResult EvaluateRegistration(
        const PointCloud &source,
        const PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d
                &transformation /* = Eigen::Matrix4d::Identity()*/) {
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    PointCloud pcd = source;
    if (transformation.isIdentity() == false) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
}

RegistrationResult RegistrationICP(
        const PointCloud &source,
        const PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        PrintError("Error: Invalid max_correspondence_distance.\n");
        return RegistrationResult(init);
    }
    if (estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::PointToPlane &&
        (!source.HasNormals() || !target.HasNormals())) {
        PrintError(
                "Error: TransformationEstimationPointToPlane requires "
                "pre-computed normal vectors.\n");
        return RegistrationResult(init);
    }

    Eigen::Matrix4d transformation = init;
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    PointCloud pcd = source;
    if (init.isIdentity() == false) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        PrintDebug("ICP Iteration #%d: Fitness %.4f, RMSE %.4f\n", i,
                   result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}

RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const PointCloud &source,
        const PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 6*/,
        const RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || (int)corres.size() < ransac_n ||
        max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }
    std::srand((unsigned int)std::time(0));
    Eigen::Matrix4d transformation;
    CorrespondenceSet ransac_corres(ransac_n);
    RegistrationResult result;
    for (int itr = 0;
         itr < criteria.max_iteration_ && itr < criteria.max_validation_;
         itr++) {
        for (int j = 0; j < ransac_n; j++) {
            ransac_corres[j] = corres[std::rand() % (int)corres.size()];
        }
        transformation =
                estimation.ComputeTransformation(source, target, ransac_corres);
        PointCloud pcd = source;
        pcd.Transform(transformation);
        auto this_result = EvaluateRANSACBasedOnCorrespondence(
                pcd, target, corres, max_correspondence_distance,
                transformation);
        if (this_result.fitness_ > result.fitness_ ||
            (this_result.fitness_ == result.fitness_ &&
             this_result.inlier_rmse_ < result.inlier_rmse_)) {
            result = this_result;
        }
    }
    PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness_,
               result.inlier_rmse_);
    return result;
}

RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
        const PointCloud &source,
        const PointCloud &target,
        const Feature &source_feature,
        const Feature &target_feature,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 4*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers /* = {}*/,
        const RANSACConvergenceCriteria &criteria,
        const double ransac_random_seed, /* default: 0*/
        const int fitness_type /* 0: standard ransac fitness function. 1: use the number of inliers (instead of the ration); 2: use 1/d^2 for inliers, where d is descriptor distance.*/

        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    RegistrationResult result;
    int total_validation = 0;
    bool finished_validation = false;
    int num_similar_features = 1;
    std::vector<std::vector<int>> similar_features(source.points_.size());


#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        CorrespondenceSet ransac_corres(ransac_n);
        KDTreeFlann kdtree(target);
        KDTreeFlann kdtree_feature(target_feature);
        RegistrationResult result_private;
        unsigned int seed_number;
#ifdef _OPENMP
        // each thread has different seed_number
        //seed_number = (unsigned int)std::time(0) * (omp_get_thread_num() + 1);
        seed_number = (unsigned int)ransac_random_seed * (omp_get_thread_num() + 1);
#else
        //seed_number = (unsigned int)std::time(0);
        seed_number = (unsigned int)ransac_random_seed;//std::time(0);
#endif
        std::srand(seed_number);

#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (!finished_validation) {
                std::vector<double> dists(num_similar_features);
                Eigen::Matrix4d transformation;
                for (int j = 0; j < ransac_n; j++) {
                    int source_sample_id =
                            std::rand() % (int)source.points_.size();
                    if (similar_features[source_sample_id].empty()) {
                        std::vector<int> indices(num_similar_features);
                        kdtree_feature.SearchKNN(
                                Eigen::VectorXd(source_feature.data_.col(
                                        source_sample_id)),
                                num_similar_features, indices, dists);
#ifdef _OPENMP
#pragma omp critical
#endif
                        { similar_features[source_sample_id] = indices; }
                    }
                    ransac_corres[j](0) = source_sample_id;
                    if (num_similar_features == 1)
                        ransac_corres[j](1) =
                                similar_features[source_sample_id][0];
                    else
                        ransac_corres[j](1) =
                                similar_features[source_sample_id]
                                                [std::rand() %
                                                 num_similar_features];
                }
                bool check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == false &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                transformation = estimation.ComputeTransformation(
                        source, target, ransac_corres);
                check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == true &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                PointCloud pcd = source;
                pcd.Transform(transformation);
                auto this_result = GetRegistrationResultAndCorrespondencesCustom(
                        pcd, target, kdtree, max_correspondence_distance,
                        transformation, source_feature, target_feature, fitness_type);
                if (this_result.fitness_ > result_private.fitness_ ||
                    (this_result.fitness_ == result_private.fitness_ &&
                     this_result.inlier_rmse_ < result_private.inlier_rmse_)) {
                    result_private = this_result;
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    total_validation = total_validation + 1;
                    if (total_validation >= criteria.max_validation_)
                        finished_validation = true;
                }
            }  // end of if statement
        }      // end of for-loop
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (result_private.fitness_ > result.fitness_ ||
                (result_private.fitness_ == result.fitness_ &&
                 result_private.inlier_rmse_ < result.inlier_rmse_)) {
                result = result_private;
            }
        }
#ifdef _OPENMP
    }
#endif
    PrintDebug("total_validation : %d\n", total_validation);
    PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness_,
               result.inlier_rmse_);
    return result;
}

RegistrationResult RegistrationRANSACBasedOnShapeComplementarity(
        const PointCloud &source,
        const PointCloud &target,
        const TriangleMesh &target_mesh,
        const PointCloud &target_face_centroids_pcd,
        const Feature &source_feature,
        const Feature &target_feature,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 4*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers /* = {}*/,
        const RANSACConvergenceCriteria &criteria,
        const double ransac_random_seed, /* default: 0*/
        const int fitness_type /* 0: standard ransac fitness function. 1: use the number of inliers (instead of the ration); 2: use 1/d^2 for inliers, where d is descriptor distance.*/

        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    RegistrationResult result;
    int total_validation = 0;
    bool finished_validation = false;
    int num_similar_features = 1;
    std::vector<std::vector<int>> similar_features(source.points_.size());


#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        CorrespondenceSet ransac_corres(ransac_n);
        // Do a KDTree for the triangles.
        PointCloud target_triangle_centroids();
        KDTreeFlann kdtree(target_face_centroids_pcd);
        KDTreeFlann kdtree_feature(target_feature);
        RegistrationResult result_private;
        unsigned int seed_number;
#ifdef _OPENMP
        // each thread has different seed_number
        //seed_number = (unsigned int)std::time(0) * (omp_get_thread_num() + 1);
        seed_number = (unsigned int)ransac_random_seed * (omp_get_thread_num() + 1);
#else
        //seed_number = (unsigned int)std::time(0);
        seed_number = (unsigned int)ransac_random_seed;//std::time(0);
#endif
        std::srand(seed_number);

#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (!finished_validation) {
                std::vector<double> dists(num_similar_features);
                Eigen::Matrix4d transformation;
                for (int j = 0; j < ransac_n; j++) {
                    int source_sample_id =
                            std::rand() % (int)source.points_.size();
                    if (similar_features[source_sample_id].empty()) {
                        std::vector<int> indices(num_similar_features);
                        kdtree_feature.SearchKNN(
                                Eigen::VectorXd(source_feature.data_.col(
                                        source_sample_id)),
                                num_similar_features, indices, dists);
#ifdef _OPENMP
#pragma omp critical
#endif
                        { similar_features[source_sample_id] = indices; }
                    }
                    ransac_corres[j](0) = source_sample_id;
                    if (num_similar_features == 1)
                        ransac_corres[j](1) =
                                similar_features[source_sample_id][0];
                    else
                        ransac_corres[j](1) =
                                similar_features[source_sample_id]
                                                [std::rand() %
                                                 num_similar_features];
                }
                bool check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == false &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                transformation = estimation.ComputeTransformation(
                        source, target, ransac_corres);
                check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == true &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                PointCloud pcd = source;
                pcd.Transform(transformation);
                //PrintInfo("###################\n"); 
                //PrintInfo("###################\n"); 
                //PrintInfo("Evaluating transformation\n"); 
                auto this_result = GetRegistrationResultAndCorrespondencesShapeComplementarity(
                        pcd, target_mesh, kdtree, max_correspondence_distance,
                        transformation, source_feature, target_feature, fitness_type);
                if (this_result.fitness_ > result_private.fitness_ ||
                    (this_result.fitness_ == result_private.fitness_ &&
                     this_result.inlier_rmse_ < result_private.inlier_rmse_)) {
                    result_private = this_result;
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    total_validation = total_validation + 1;
                    if (total_validation >= criteria.max_validation_)
                        finished_validation = true;
                }
            }  // end of if statement
        }      // end of for-loop
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (result_private.fitness_ > result.fitness_ ||
                (result_private.fitness_ == result.fitness_ &&
                 result_private.inlier_rmse_ < result.inlier_rmse_)) {
                result = result_private;
            }
        }
#ifdef _OPENMP
    }
#endif
    PrintDebug("total_validation : %d\n", total_validation);
    PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness_,
               result.inlier_rmse_);
    return result;
}

Eigen::Matrix6d GetInformationMatrixFromPointClouds(
        const PointCloud &source,
        const PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    PointCloud pcd = source;
    if (transformation.isIdentity() == false) {
        pcd.Transform(transformation);
    }
    RegistrationResult result;
    KDTreeFlann target_kdtree(target);
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, target_kdtree, max_correspondence_distance,
            transformation);

    // write q^*
    // see http://redwood-data.org/indoor/registration.html
    // note: I comes first in this implementation
    Eigen::Matrix6d GTG = Eigen::Matrix6d::Identity();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Identity();
        Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (auto c = 0; c < result.correspondence_set_.size(); c++) {
            int t = result.correspondence_set_[c](1);
            double x = target.points_[t](0);
            double y = target.points_[t](1);
            double z = target.points_[t](2);
            G_r_private.setZero();
            G_r_private(1) = z;
            G_r_private(2) = -y;
            G_r_private(3) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = -z;
            G_r_private(2) = x;
            G_r_private(4) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = y;
            G_r_private(1) = -x;
            G_r_private(5) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        { GTG += GTG_private; }
#ifdef _OPENMP
    }
#endif
    return std::move(GTG);
}

}  // namespace open3d
