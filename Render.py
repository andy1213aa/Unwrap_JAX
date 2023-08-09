from Camera import Camera
from KDtree import KDTree
import time
import numpy as np
from Ray import Ray
import utlis

class Renderer:
    '''
    Class responsible for rendering an image.
    '''

    def __init__(self, camera: Camera, kd_tree: KDTree, ray: Ray):
        self.camera = camera
        self.kd_tree = kd_tree
        self.rays = ray

        
    def inverse_ray_trace(self, rays):

        color = np.zeros((len(rays), 3))
        t = 0
        not_occlude = 0

        for i, ray in enumerate(rays):
            start = time.time()

            ret = self.kd_tree.intersect(i, ray)
            duration = time.time() - start
            t += duration
            if not ret:  # No occlude
                plane_intersect = self.camera.get_plane_intersect(
                    ray.origin,
                    ray.direction,
                )

                if (-self.camera.princpt_mm[0] < plane_intersect[0]) and (
                        -self.camera.princpt_mm[0] + self.camera.width_mm >
                        plane_intersect[0]
                ) and (-self.camera.princpt_mm[1] < plane_intersect[1]) and (
                        -self.camera.princpt_mm[1] + self.camera.height_mm >
                        plane_intersect[1]):
                    not_occlude += 1
                    pixel_xy = self.camera.xy2pixel(plane_intersect[:2])
                    color[i] = np.array([255, 0, 0])
                    # color[i] = self.camera.img[pixel_xy[1]][pixel_xy[0]]

        print(f'Total Extime: {t}s')
        print(f'No occlude: {not_occlude}')

        return color
    
    @utlis.measureExcutionTime
    def ray_tracing(self, rays):
        
        result = []
        for i, ray in enumerate(rays):
            ret = self.kd_tree.intersect(i, ray, return_nearly=True)
            if ret:
                result += ret
 
        print(f'occlude: {len(result)}')
        return result
                
    def render_texel(self, ):

        image = self.inverse_ray_trace(self.rays)
        return image

    def feature_marking(self,):
        
        result = self.ray_tracing(self.rays)
        return result
    
    # def tarce_collision_ray(self, ray:Ray):
    #     pass