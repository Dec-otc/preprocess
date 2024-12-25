import os

import numpy as np
import open3d as o3d


def euclidean_cluster(cloud, tolerance=0.2, min_cluster_size=100, max_cluster_size=1000):
    """
    欧式聚类
    :param cloud:输入点云
    :param tolerance: 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
    :param min_cluster_size:设置一个聚类需要的最少的点数目
    :param max_cluster_size:设置一个聚类需要的最大点数目
    :return:聚类个数
    """

    kdtree = o3d.geometry.KDTreeFlann(cloud)  # 对点云建立kd树索引

    num_points = len(cloud.points)
    processed = [-1] * num_points  # 定义所需变量
    clusters = []  # 初始化聚类
    # 遍历各点
    for idx in range(num_points):
        if processed[idx] == 1:  # 如果该点已经处理则跳过
            continue
        seed_queue = []  # 定义一个种子队列
        sq_idx = 0
        seed_queue.append(idx)  # 加入一个种子点
        processed[idx] = 1

        while sq_idx < len(seed_queue):

            k, nn_indices, _ = kdtree.search_radius_vector_3d(cloud.points[seed_queue[sq_idx]], tolerance)

            if k == 1:  # k=1表示该种子点没有近邻点
                sq_idx += 1
                continue
            for j in range(k):

                if nn_indices[j] == num_points or processed[nn_indices[j]] == 1:
                    continue  # 种子点的近邻点中如果已经处理就跳出此次循环继续
                seed_queue.append(nn_indices[j])
                processed[nn_indices[j]] = 1

            sq_idx += 1

        if max_cluster_size > len(seed_queue) > min_cluster_size:
            clusters.append(seed_queue)

    return clusters


if __name__ == '__main__':

    DATA = "Test3"
    INPUT_PATH = f"D:\\fff-20240516\\Data\\{DATA}.pcd"
    OUTPUT_PATH = f"D:\\fff-20240516\\Result\\{DATA}\\Euclidean\\"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --------------------------加载点云数据------------------------------
    pcd = o3d.io.read_point_cloud(INPUT_PATH)
    # print(np.array(pcd.points).shape)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    print(np.array(pcd.points).shape)

    # pcd.points = o3d.utility.Vector3dVector(
    #     np.asarray(pcd.points) * 2 / np.max(np.abs(np.asarray(pcd.points))))

    # o3d.visualization.draw_geometries([pcd], window_name="原始点云",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)

    # ---------------------------欧式聚类--------------------------------
    # Test1
    ec = euclidean_cluster(pcd,
                           tolerance=0.02,      # 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
                           min_cluster_size=40, # 设置一个聚类需要的最少的点数目
                           max_cluster_size=100000) # 设置一个聚类需要的最大点数目
    # Test2
    ec = euclidean_cluster(pcd,
                           tolerance=0.14,         # 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
                           min_cluster_size=85,    # 设置一个聚类需要的最少的点数目
                           max_cluster_size=100000)# 设置一个聚类需要的最大点数目



    # Test3
    ec = euclidean_cluster(pcd, tolerance=0.2,    # 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
                           min_cluster_size=500,  # 设置一个聚类需要的最少的点数目
                           max_cluster_size=100000) # 设置一个聚类需要的最大点数目

    # Test4
    ec = euclidean_cluster(pcd, tolerance=0.45,          # 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
                           min_cluster_size=50,          # 设置一个聚类需要的最少的点数目
                           max_cluster_size=100000)      # 设置一个聚类需要的最大点数目
    # -------------------------聚类结果分类保存---------------------------
    print("聚类个数：", len(ec))
    for i in range(len(ec)):
        ind = ec[i]
        clusters_cloud = pcd.select_by_index(ind)
        file_name = OUTPUT_PATH + "euclidean_cluster" + str(i + 1) + ".pcd"
        o3d.io.write_point_cloud(file_name, clusters_cloud)
    merged_pcd = o3d.geometry.PointCloud()
    segment = []  # 存储分割结果的容器
    for i in range(len(ec)):
        ind = ec[i]
        clusters_cloud = pcd.select_by_index(ind)
        r_color = np.random.uniform(0, 1, (1, 3))  # 分类点云随机赋色
        clusters_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        segment.append(clusters_cloud)
        merged_pcd += clusters_cloud
    o3d.io.write_point_cloud(OUTPUT_PATH+"out.pcd", merged_pcd)

    # -----------------------------结果可视化------------------------------------
    o3d.visualization.draw_geometries(segment, window_name="",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)