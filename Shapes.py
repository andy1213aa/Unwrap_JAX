from AABB import AABB
import numpy as np

mini = 1e+6
class Triangle():
    # Möller–Trumbore ray-triangle intersection algorithm
    def __init__(self, idx, vertices):
        self.idx = idx
        self.vertices = vertices
        self.aabb = AABB(np.min(vertices, axis=0), np.max(vertices, axis=0))
        # print('-----------------')
        # print(f'Vertices: {self.vertices}')
        # print(f'AABB: [{self.aabb.min_corner, self.aabb.max_corner}]')
        
        # global mini
        # if self.aabb.min_corner[0] < mini:
        #     mini = self.aabb.min_corner[0]
        #     print(mini)
        
    def intersect(self, ray_origin, ray_direction):
        epsilon = 1e-6
        v0, v1, v2 = self.vertices

        edge1 = v1 - v0
        edge2 = v2 - v0
        pvec = np.cross(ray_direction, edge2)
        det = np.dot(edge1, pvec)

        if abs(det) < epsilon:
            return False

        inv_det = 1.0 / det
        tvec = ray_origin - v0
        u = np.dot(tvec, pvec) * inv_det

        if u < 0. or u > 1.:
            return False

        qvec = np.cross(tvec, edge1)
        v = np.dot(ray_direction, qvec) * inv_det

        if v < 0. or u + v > 1.:
            return False

        t = np.dot(edge2, qvec) * inv_det

        if t < epsilon:
            return False

        
        return True
