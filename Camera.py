import numpy as np


class Camera:
    '''
    A class for keeping track of camera parameters.
    '''

    def __init__(
        self,
        origin,
        width: int,
        height: int,
        K,
        R,
        t,
        pixel_size_mm,
        focal: float,
        princpt: float,
        
        img: np.array,
    ):
        self.origin = origin
        self.width = width
        self.height = height
        self.K = K
        self.R = R
        self.t = t
        self.focal = focal
        self.princpt = princpt
        self.normal_vector = np.array([0, 0, 1])
        self.img = img
        
        # 像素的物理尺寸
        self.pixel_size_mm = pixel_size_mm#0.00345  # mm
        # 像素焦距，這些數值通常可以從內部參數矩陣中獲得
        fx_pixels = focal[0]
        fy_pixels = focal[1]

        # 主點座標，這些數值通常可以從內部參數矩陣中獲得
        cx_pixels = princpt[0]
        cy_pixels = princpt[1]

        # 轉換為物理焦距
        f_x_mm = fx_pixels * self.pixel_size_mm
        f_y_mm = fy_pixels * self.pixel_size_mm
        self.focal_mm = np.array([f_x_mm, f_y_mm])
        
        # 轉換為物理座標
        cx_mm = cx_pixels * self.pixel_size_mm
        cy_mm = cy_pixels * self.pixel_size_mm
        self.princpt_mm = np.array([cx_mm, cy_mm])

        self.width_mm = width * self.pixel_size_mm
        self.height_mm = height * self.pixel_size_mm
        
        print(f'self.focal_mm: {self.focal_mm}')
        print(f'self.princpt_mm: {self.princpt_mm}')
        
    
    def xy2pixel(self, plane_xy):

        # shift coordinate
        plane_xy += np.array(
            [self.princpt_mm[0], self.princpt_mm[1]])

        pixel_xy = np.zeros(plane_xy.shape)

        pixel_xy[0] = plane_xy[0] / self.pixel_size_mm
        pixel_xy[1] = plane_xy[1] / self.pixel_size_mm

        #return unit: pixel
        return pixel_xy.astype(np.int32)
    
    def pixel2xy(self, pixel_xy):
        
        # convert pixel to coordinate in mm
        plane_xy = np.zeros((2,))

        plane_xy[0] = pixel_xy[0] * self.pixel_size_mm
        plane_xy[1] = pixel_xy[1] * self.pixel_size_mm

        # shift coordinate
        plane_xy -= np.array(
            [self.princpt_mm[0], self.princpt_mm[1]])

        # return unit: coordinate in xy plane
        return plane_xy
    
    def get_plane_intersect(self, ray_origin, ray_dir):

        # Ax+By+Cz+D=0
        normal_vector = self.normal_vector  #(A, B, C)
        D = -self.focal_mm[0]  # D

        # 求交點

        t = -(D + np.dot(normal_vector, ray_origin)) / np.dot(
            normal_vector, ray_dir)

        intersection_point = ray_origin + t * ray_dir
        
        # return unit: mm
        return intersection_point