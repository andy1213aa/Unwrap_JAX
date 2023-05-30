import jax.numpy as jnp
from Ray import Ray


class triangle():
    '''
    Class for a triangle defined as three vertexes.
    '''

    def __init__(self, vertexes: jnp.array):
        self.vertex_x = vertexes[0]
        self.vertex_y = vertexes[1]
        self.vertex_z = vertexes[2]
        self.vertexes = vertexes

    def intersect(
        self,
        ray_origin,
        ray_direction,
    ):
        # 计算边缘向量和法线
        edge1 = self.vertex_y - self.vertex_x 
        edge2 = self.vertex_z - self.vertex_x 
        triangle_normal = jnp.cross(edge1, edge2)

        # 计算射线和平面的交点
        epsilon = 1e-6  # 用于避免除零错误的小数
        denom = jnp.dot(ray_direction, triangle_normal)
        t = jnp.dot(self.vertex_x  - ray_origin, triangle_normal) / max(
            denom, epsilon)

        if t < 0:
            return False  # 交点在射线之前，不相交

        # 计算交点的坐标
        intersection_point = ray_origin + t * ray_direction

        # 检查交点是否在三角形内部
        edge0 = self.vertex_x  - self.vertex_z
        edge1 = self.vertex_y - self.vertex_z
        C = intersection_point - self.vertex_z

        u = jnp.dot(jnp.cross(edge1, C), triangle_normal)
        v = jnp.dot(jnp.cross(C, edge0), triangle_normal)

        return (u >= 0) and (v >= 0) and (u + v <= 1)
