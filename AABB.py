class AABB:

    def __init__(self, min_corner, max_corner):
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.bounds = [self.min_corner, self.max_corner]

    def intersect(self, ray_origin, ray_direction):
        sign = [0, 0, 0]  # Whether each ray smaller then 0

        sign[0] = (ray_direction[0] < 0)
        sign[1] = (ray_direction[1] < 0)
        sign[2] = (ray_direction[2] < 0)
        invdir = 1 / ray_direction

        t_min = (self.bounds[sign[0]][0] - ray_origin[0]) * invdir[0]
        t_max = (self.bounds[1 - sign[0]][0] - ray_origin[0]) * invdir[0]

        ty_min = (self.bounds[sign[1]][1] - ray_origin[1]) * invdir[1]
        ty_max = (self.bounds[1 - sign[1]][1] - ray_origin[1]) * invdir[1]

        
        if (t_min > ty_max) or (ty_min > t_max):
            return False

        if ty_min > t_min:
            t_min = ty_min

        if ty_max < t_max:
            t_max = ty_max

        tz_min = (self.bounds[sign[2]][2] - ray_origin[2]) * invdir[2]
        tz_max = (self.bounds[1 - sign[2]][2] - ray_origin[2]) * invdir[2]

        if (t_min > tz_max) or (tz_min > t_max):
            return False
        
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        return True