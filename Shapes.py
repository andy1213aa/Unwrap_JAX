from AABB import AABB
import numpy as np


class Triangle:

    def __init__(self, vertices):
        self.vertices = vertices

        self.aabb = AABB(np.min(vertices, axis=0), np.max(vertices, axis=0))

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