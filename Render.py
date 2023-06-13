import jax
from functools import reduce
from Camera import Camera
from Ray import Ray
from Vec3 import Vec3
from KDtree import KDTree
import time
import numpy as np
from Scene import Scene
import cv2


class Renderer:
    '''
    Class responsible for rendering an image.
    '''

    def __init__(self, camera: Camera, kd_tree: KDTree, ray_dir: np.ndarray):
        self.camera = camera
        self.kd_tree = kd_tree
        self.ray_dir = ray_dir

    def xy2pixel(self, plane_xy):

        # shift coordinate
        plane_xy += np.array(
            [self.camera.princpt_mm[0], self.camera.princpt_mm[1]])

        pixel_xy = np.zeros(plane_xy.shape)

        pixel_xy[0] = plane_xy[0] / self.camera.pixel_size_mm
        pixel_xy[1] = plane_xy[1] / self.camera.pixel_size_mm

        return pixel_xy.astype(np.int32)

    def get_plane_intersect(self, ray_origin, ray_dir):

        # Ax+By+Cz+D=0
        normal_vector = self.camera.normal_vector  #(A, B, C)
        D = -self.camera.focal_mm[0]  # D

        # 求交點

        t = -(D + np.dot(normal_vector, ray_origin)) / np.dot(
            normal_vector, ray_dir)
        
        intersection_point = ray_origin + t * ray_dir
        return intersection_point

    def ray_trace(self, origin, normalized_direction):

        color = np.zeros(normalized_direction.shape)
        t = 0
        not_occlude = 0

        for i, norm_ray_dir in enumerate(normalized_direction):
            start = time.time()

            ret = self.kd_tree.intersect(i, origin[i], norm_ray_dir)
            duration = time.time() - start
            t += duration
            if not ret:  # No occlude
                plane_intersect = self.get_plane_intersect(
                    origin[i],
                    norm_ray_dir,
                )
                
                if (-self.camera.princpt_mm[0] < plane_intersect[0]) and (
                        -self.camera.princpt_mm[0] + self.camera.width_mm >
                        plane_intersect[0]
                ) and (-self.camera.princpt_mm[1] < plane_intersect[1]) and (
                        -self.camera.princpt_mm[1] + self.camera.height_mm >
                        plane_intersect[1]):
                    
                    not_occlude += 1
                    pixel_xy = self.xy2pixel(plane_intersect[:2])

                    color[i] = self.camera.img[pixel_xy[1]][pixel_xy[0]]

        print(f'Total Extime: {t}s')
        print(f'No occlude: {not_occlude}')

        return color

    def render_texel(
        self,
        ray_origin,
    ):

        image = self.ray_trace(
            ray_origin,
            self.ray_dir,
        )
        return image
