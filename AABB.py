class AABB:
    def __init__(self, min_corner, max_corner):
        self.min_corner = min_corner
        self.max_corner = max_corner

    def intersect(self, ray_origin, ray_direction):
        epsilon = 1e-6  # 極小值
        t_min = (self.min_corner[0] - ray_origin[0]) / (ray_direction[0] + epsilon)
        t_max = (self.max_corner[0] - ray_origin[0]) / (ray_direction[0] + epsilon)

        if t_min > t_max:
            t_min, t_max = t_max, t_min

        ty_min = (self.min_corner[1] - ray_origin[1]) / (ray_direction[1] + epsilon)
        ty_max = (self.max_corner[1] - ray_origin[1]) / (ray_direction[1] + epsilon)

        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

        if (t_min > ty_max) or (ty_min > t_max):
            return False

        if ty_min > t_min:
            t_min = ty_min

        if ty_max < t_max:
            t_max = ty_max

        tz_min = (self.min_corner[2] - ray_origin[2]) / (ray_direction[2] + epsilon)
        tz_max = (self.max_corner[2] - ray_origin[2]) / (ray_direction[2] + epsilon)

        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        if (t_min > tz_max) or (tz_min > t_max):
            return False

        return True