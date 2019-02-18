#include <iostream>
#include "file-system-tools.h"
#include <algorithm>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>

// align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, Eigen::Matrix4f& est_traansformation) {
    registration->setInputTarget(target_cloud);
    registration->setInputSource(source_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

    auto t1 = ros::WallTime::now();
    registration->align(*aligned);
    auto t2 = ros::WallTime::now();
    std::cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;

    for(int i=0; i<10; i++) {
        registration->align(*aligned);
    }
    auto t3 = ros::WallTime::now();

    est_traansformation = registration->getFinalTransformation();

    std::cout << "10times: " << (t3 - t2).toSec() * 1000 << "[msec]" << std::endl;
    std::cout << "fitness: " << registration->getFitnessScore() << std::endl << std::endl;
    std::cout << "transfrom: \n" << est_traansformation << std::endl << std::endl;


    return aligned;
}

int main(int argc, char ** argv) {

    std::string dataset_path = "/home/pang/data/dataset/kitti/00";
    std::string depth_from_image_folder = dataset_path + "/image_depth_pcd";
    std::string depth_from_lidar_folder = dataset_path + "/lidar_pcd";

    /// image pcd
    std::vector<std::string> image_pcd_vector, lidar_pcd_vector;
    common::getAllFilesInFolder(depth_from_image_folder, &image_pcd_vector);
    std::cout<<"image_pcd_vector: " << image_pcd_vector.size() << std::endl;

    std::sort(image_pcd_vector.begin(),image_pcd_vector.end(), [](std::string a, std::string b) {
        return !common::compareNumericPartsOfStrings(a,b);
    });

    common::getAllFilesInFolder(depth_from_lidar_folder, &lidar_pcd_vector);
    std::cout<<"lidar_pcd_vector: " << lidar_pcd_vector.size() << std::endl;

    std::sort(lidar_pcd_vector.begin(),lidar_pcd_vector.end(), [](std::string a, std::string b) {
        return !common::compareNumericPartsOfStrings(a,b);
    });

    ros::Time::init();

    for (int i = 0; i < lidar_pcd_vector.size(); i++ ) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        std::string target_pcd = lidar_pcd_vector[i];
        std::string source_pcd = image_pcd_vector[i];

        if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
            std::cerr << "failed to load " << target_pcd << std::endl;
            return 0;
        }
        if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
            std::cerr << "failed to load " << source_pcd << std::endl;
            return 0;
        }

        /// try to align
        // downsampling
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

        voxelgrid.setInputCloud(target_cloud);
        voxelgrid.filter(*downsampled);
        *target_cloud = *downsampled;

        voxelgrid.setInputCloud(source_cloud);
        voxelgrid.filter(*downsampled);
        source_cloud = downsampled;

        pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
        ndt_omp->setResolution(1.0);


        ndt_omp->setNumThreads(8);
        ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        Eigen::Matrix4f est_transform;
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned = align(ndt_omp, target_cloud, source_cloud, est_transform);




    }


    return 0;
}