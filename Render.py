import jax
from functools import reduce
from Camera import Camera
from Ray import Ray
from Vec3 import Vec3
from KDtree import KDTree
import time
import numpy as np
from Scene import Scene


class Renderer:
    '''
    Class responsible for rendering an image.
    '''

    def __init__(self, camera: Camera, kd_tree: KDTree, ray_dir: np.ndarray):
        self.camera = camera
        self.kd_tree = kd_tree
        self.ray_dir = ray_dir

    def ray_color(self, r: Ray):
        '''
        Evaluates the color of a ray based on where it hits the background.
        '''
        unit_direction = r.ray_direction / np.linalg.norm(r.ray_direction)
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1., 1., 1.]) + t * np.array(
            [0.5, 0.7, 1.0])

    # def render(self):
    #     image = np.zeros((self.camera.height, self.camera.width, 3))
    #     for j in range(self.camera.height - 1, -1, -1):
    #         for i in range(self.camera.width):
    #             u = float(i) / (self.camera.width - 1)
    #             v = float(j) / (self.camera.height - 1)
    #             r = Ray(
    #                 self.camera.origin, self.camera.lower_left_corner +
    #                 u * self.camera.horizontal + v * self.camera.vertical -
    #                 self.camera.origin)
    #             image = jax.ops.index_update(image, jax.ops.index[j, i, :],
    #                                          self.ray_color(r))
    #     return image

    def xy2pixel(self, plane_xy):

        # shift coordinate
        plane_xy += np.array(
            [self.camera.princpt_mm[0], self.camera.princpt_mm[1]])

        pixel_xy = np.zeros(plane_xy.shape)
        x_interval = self.camera.princpt[0] * 2 / self.camera.width
        y_interval = self.camera.princpt[1] * 2 / self.camera.height

        pixel_xy[0] = plane_xy[0] // x_interval
        pixel_xy[1] = plane_xy[1] // y_interval

        return pixel_xy.astype(np.int32)

    def get_plane_intersect(self, ray_origin, ray_dir):

        # Ax+By+Cz+D=0
        normal_vector = self.camera.normal_vector  #(A, B, C)
        D = -self.camera.focal_mm[0]  # D

        # 求交點

        t = -(D + np.dot(normal_vector, ray_origin)) / np.dot(
            normal_vector, ray_dir)

        # print(f't*ray: {(t * ray_dir).shape}')
        intersection_point = ray_origin + t * ray_dir
        return intersection_point

    def ray_trace(self, origin, normalized_direction):
        # far = 1.0e15  # A large number, which we can never hit
        # distances = [
        #     o.ray_triangle_intersection(origin, normalized_direction)
        #     for o in self.scene.objects
        # ]
        color = np.zeros(normalized_direction.shape)
        t = 0
        not_occlude = 0
        text = ''
        for i, norm_ray_dir in enumerate(normalized_direction):
            start = time.time()
            ret, ret_AABB = self.kd_tree.intersect(origin, norm_ray_dir)
            duration = time.time() - start
            # print(self.camera.princpt)
            # canvas_width = self.
            info = '------------------ \n' + f'ith {i} \n' + f'ret: {len(ret)} \n' + f'ret_AABB: {len(ret_AABB)} \n' + '------------------\n'
            text += info
            # print('------------------')
            # print(f'ith {i}')
            # print(f'ret: {len(ret)}')
            # print(f'ret_AABB: {len(ret_AABB)}')
            # print('------------------')
            if not ret_AABB:  # No occlude
                plane_intersect = self.get_plane_intersect(
                    origin,
                    norm_ray_dir,
                )

                plane_intersect2 = norm_ray_dir * self.camera.focal_mm[
                    0] / norm_ray_dir[2]

                # print(f'plane_intersect: {plane_intersect}')
                # print(f'plane_intersect: {plane_intersect}')
                width_mm = self.camera.width * self.camera.pixel_size_mm
                height_mm = self.camera.height * self.camera.pixel_size_mm

                # if plane_intersect[0] > -self.camera.princpt_mm[
                #         0] and plane_intersect[0] < (
                #             -self.camera.princpt_mm[0] + width_mm
                #         ) and plane_intersect[1] > -self.camera.princpt_mm[
                #             1] and plane_intersect[1] < (
                #                 -self.camera.princpt_mm[1] + height_mm):
                # not_occlude += 1
                # if (-width_mm / 2 < plane_intersect[0]) and (
                #         width_mm / 2 > plane_intersect[0]) and (
                #             -height_mm / 2 < plane_intersect[1]) and (
                #                 height_mm / 2 > plane_intersect[1]):

                color[i] = np.array([255, 0, 0])
                # pixel_xy = self.xy2pixel(plane_intersect[:2])
                # print(f'pixel_xy: {pixel_xy}')
                # color[i] = self.camera.img[pixel_xy[1]][pixel_xy[0]]
                # print(color.shape)

            t += duration

            # print(f'{i}th Extime: {duration}s')
        print(f'log: {text}')
        with open('log.txt', 'w') as f:
            f.write(text)
        print(f'Total Extime: {t}s')
        print(f'No occlude: {not_occlude}')

        # nearest = reduce(np.minimum, distances)

        # for (o, d) in zip(self.scene.objects, distances):
        #     color += o.light(
        #         origin, normalized_direction, d, self.light_position, origin,
        #         self.scene.objects) * (nearest != far) * (d == nearest)
        print(f'color shape: {color.shape}')
        print(f'color: {color}')
        return color

    def render_texel(self):
        # r = float(self.camera.width) / self.camera.height
        # # S = (-1., 1. / r + .25, 1., -1. / r + .25)
        # S = (
        #     -self.camera.princpt[0],
        #     -self.camera.princpt[1],
        #     self.camera.princpt[0],
        #     self.camera.princpt[1],
        # )
        # x = np.tile(
        #     np.linspace(S[0], S[2], self.camera.width),
        #     self.camera.height,
        # )
        # y = np.repeat(
        #     np.linspace(S[1], S[3], self.camera.height),
        #     self.camera.width,
        # )

        # z = np.repeat(np.array([self.camera.focal[0]]), y.shape[0])

        # print('Canvas Border:')
        # print(f'X: [{np.min(x)}, {np.max(x)}]')
        # print(f'Y: [{np.min(y)}, {np.max(y)}]')
        # print(f'Z: [{np.min(z)}, {np.max(z)}]')

        image = self.ray_trace(
            self.camera.origin,
            self.ray_dir,
        )
        return image
