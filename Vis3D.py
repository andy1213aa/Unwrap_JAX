import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import combinations


class Vis_tracing_3D():

    def __init__(self, verts, used_vtx_idx):
        # 创建3D图形对象

        self.verts = verts
        self.used_vtx_idx = used_vtx_idx

    def draw_aabb_tracing(self, ith, ray_origin, aabb):
        fig = plt.figure(figsize=(50, 20))
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=-38., azim=-89)

        color = ['black'] * 153555
        size = [1] * 153555
        color[ith] = 'r'
        size[ith] = 50
        # 创建示例数据
        x = self.verts[:, 0]
        y = self.verts[:, 1]
        z = self.verts[:, 2]
        # for i in self.used_vtx_idx:
        #     color[i] = 'b'

        # 绘制3D点
        self.ax.scatter(x, y, z, s=size, c=color)
        x = [ray_origin[0], 0]
        y = [ray_origin[1], 0]
        z = [ray_origin[2], 0]

        self.ax.plot(x, y, z, 'g')

        # 繪製AABB

        corner1 = aabb.min_corner
        corner2 = np.array(
            [aabb.max_corner[0], aabb.min_corner[1], aabb.min_corner[2]])
        corner3 = np.array(
            [aabb.min_corner[0], aabb.max_corner[1], aabb.min_corner[2]])
        corner4 = np.array(
            [aabb.min_corner[0], aabb.min_corner[1], aabb.max_corner[2]])
        corner5 = np.array(
            [aabb.max_corner[0], aabb.max_corner[1], aabb.min_corner[2]])
        corner6 = np.array(
            [aabb.min_corner[0], aabb.max_corner[1], aabb.max_corner[2]])
        corner7 = np.array(
            [aabb.max_corner[0], aabb.min_corner[1], aabb.max_corner[2]])
        corner8 = aabb.max_corner

        corners = [
            corner1, corner2, corner3, corner4, corner5, corner6, corner7,
            corner8
        ]
        # 创建一个数组
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        # 选择不重复的元素组合
        combination = np.array(list(combinations(arr, 2)))

        for edge in combination:

            vertex1 = corners[edge[0]]
            vertex2 = corners[edge[1]]

            x = [vertex1[0], vertex2[0]]
            y = [vertex1[1], vertex2[1]]
            z = [vertex1[2], vertex2[2]]

            self.ax.plot(x, y, z, 'r')
        # x = np.tile(np.linspace(-camera.princpt[1], camera.princpt[1], camera.width), camera.height)
        # y = np.tile(np.linspace(-camera.princpt[0], camera.princpt[0], camera.height), camera.width)
        # z = np.repeat(np.array([camera.focal[0]]), y.shape[0])
        # ax.scatter(y, x, z, c='g')

        # 设置坐标轴标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # plt.savefig(f'{ith}_aabb.png')
        # 显示图形
        plt.title(f'RAY {ith}')
        plt.show()
        return plt
