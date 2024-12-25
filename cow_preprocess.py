import numpy as np
import open3d as o3d
import glob
import os
from Euclidean.euclidean import euclidean_cluster
import cv2

# Camera intrinsic parameters, used to project the point cloud onto the image
f_x = 1066.09
f_y = 1066.13
c_x = 1128.79
c_y = 633.1320

def input_filter(pcd):
    # Convert the point cloud to a NumPy array
    ori_points = np.asarray(pcd.points)
    # Select part of the points in the point cloud based on filtering conditions
    indices = np.where((ori_points[:, 0] >= -0.48) & (ori_points[:, 0] <= 0.50) & \
              (ori_points[:, 1] >= -0.65) & (ori_points[:, 1] <= 0.54) & (ori_points[:, 2] >= -1.5))[0]
    # Select points by index
    cloud_filtered = pcd.select_by_index(indices)
    return cloud_filtered  # Return the filtered point cloud

def project_image(input_cloud):
    # Separate the x, y, z coordinates of the point cloud
    cloud_x, cloud_y, cloud_z = input_cloud[:, 0], input_cloud[:, 1], input_cloud[:, 2]
    # Calculate the projection coordinates of the points on the image
    image_points_x = ((f_x * -cloud_x / cloud_z) + c_x) / 2208
    image_points_y = ((f_y * cloud_y / cloud_z) + c_y) / 1242
    # Determine the width and height of the image
    image_width = int(2208 / 2)
    image_height = int(1242 / 2)
    # Create a blank image
    image = np.zeros((image_height, image_width), dtype=np.uint8)
    # Project the point cloud onto the image
    for i in range(len(image_points_x)):
        x = int(image_points_x[i] * image_width)
        y = int(image_points_y[i] * image_height)
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x] = 255  # Set the color of the projection point to white
    # Display the image
    # cv2.imshow('Projected Point Cloud', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

if __name__ == "__main__":
    index = os.listdir("F:\\test")  # Main directory for data files
    index.sort(key=lambda x: int(x))
    for id in index:  # Iterate through files with different indices
        MAIN_PATH = f"F:\\test\\{id}"
        COW_FILTER_PATH = os.path.join(MAIN_PATH, 'cow_filter_pcd')
        IMAGE_PATH = os.path.join(MAIN_PATH, 'cow_image')
        COW_0_PATH = os.path.join(MAIN_PATH, 'cow_z0_pcd')

        # Create directories if they do not exist
        os.makedirs(COW_FILTER_PATH, exist_ok=True)
        os.makedirs(IMAGE_PATH, exist_ok=True)
        os.makedirs(COW_0_PATH, exist_ok=True)

        for path in glob.glob(os.path.join(MAIN_PATH, '*.pcd')):
            pcd = o3d.io.read_point_cloud(path)  # Source point cloud file
            print(path)
            parent_dir = os.path.dirname(path)
            file_name, file_extension = os.path.splitext(os.path.basename(path))
            cloud_filtered = input_filter(pcd)  # Use passthrough filtering to get cow body point cloud in a fixed region
            # Further process the cow body point cloud using Euclidean clustering
            cloud_filtered = cloud_filtered.voxel_down_sample(voxel_size=0.005)
            ec = euclidean_cluster(cloud_filtered,
                                   tolerance=0.015,  # Set the search radius for neighbor search (i.e., minimum Euclidean distance between points of different clusters)
                                   min_cluster_size=20,  # Set the minimum number of points required for a cluster
                                   max_cluster_size=100000)  # Set the maximum number of points allowed in a cluster
            print("Number of clusters:", len(ec))
            ind = max(range(len(ec)), key=lambda i: len(ec[i]))
            clusters_cloud = cloud_filtered.select_by_index(ec[ind])  # Select the cluster with the largest number of points
            # Save the processed cow body point cloud
            filter_pcd_output = os.path.join(COW_FILTER_PATH, file_name + '.pcd')
            o3d.io.write_point_cloud(filter_pcd_output, clusters_cloud)
            print(f"ID:{id}, {file_name} filter_PCD save success!")
            # Project the cow body point cloud to get an image and save it
            project_cloud = np.asarray(clusters_cloud.points)
            image = project_image(project_cloud)
            output_path = os.path.join(IMAGE_PATH, file_name + '.png')
            cv2.imwrite(output_path, image)
            print(f"ID:{id}, {file_name} IMAGE save success!")
            # Assign the z-coordinate of the cow body point cloud to 0 to obtain parallel projection PCD
            z0_output_path = os.path.join(COW_0_PATH, file_name + '.pcd')
            z0_pcd = o3d.geometry.PointCloud()
            project_cloud[:, 2] = 0
            z0_pcd.points = o3d.utility.Vector3dVector(project_cloud)
            o3d.io.write_point_cloud(z0_output_path, z0_pcd)
            print(f"ID:{id}, {file_name} z0_PCD save success!")
