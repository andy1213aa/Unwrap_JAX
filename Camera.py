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
        self.pixel_size_mm = 0.00345  # mm
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