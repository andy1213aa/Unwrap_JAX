import jax.numpy as jnp
from Ray import Ray
import jax
from functools import partial

class triangle():
    '''
    Class for a triangle defined as three vertexes.
    '''

    def __init__(self, vertexes: jnp.array):
        self.vertex_x = vertexes[0]
        self.vertex_y = vertexes[1]
        self.vertex_z = vertexes[2]
        self.vertexes = vertexes

    # def intersect(
    #     self,
    #     ray_origin,
    #     ray_direction,
    # ):
    #     # print(f'ray_direction: {ray_direction}')
    #     # 计算边缘向量和法线
    #     edge1 = self.vertex_y - self.vertex_x 
    #     edge2 = self.vertex_z - self.vertex_x 
    #     triangle_normal = jnp.cross(edge1, edge2)

    #     # 计算射线和平面的交点
    #     epsilon = 1e-6  # 用于避免除零错误的小数
   
    #     denom = jnp.dot(ray_direction, triangle_normal)
    #     t = jnp.dot(self.vertex_x  - ray_origin, triangle_normal) / max(denom, epsilon)

    #     if t < 0:
    #         return False  # 交点在射线之前，不相交

    #     # 计算交点的坐标
    #     intersection_point = ray_origin + t * ray_direction

    #     # 检查交点是否在三角形内部
    #     edge0 = self.vertex_x  - self.vertex_z
    #     edge1 = self.vertex_y - self.vertex_z
    #     C = intersection_point - self.vertex_z

    #     u = jnp.dot(jnp.cross(edge1, C), triangle_normal)
    #     v = jnp.dot(jnp.cross(C, edge0), triangle_normal)

    #     if (u >= 0) and (v >= 0) and (u + v <= 1):
    #         print('fuck')
        
    #     return (u >= 0) and (v >= 0) and (u + v <= 1)

    def ray_triangle_intersection(self, ray_start, ray_vec):
        for ray in ray_vec:
            ret = self.intersect(ray_start, ray)
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')
        print('DONEEEEEEEEEEEEEEEEe')

    def intersect(self, ray_start, ray_vec):
        return _intersect(self.vertexes, ray_start, ray_vec)
    
@partial(jax.jit, static_argnames=['eps'])
def _intersect(vertexes, ray_start, ray_vec, eps = 0.000001):
    """Moeller–Trumbore intersection algorithm.

    Parameters
    ----------
    ray_start : np.ndarray
        Length three numpy array representing start of point.

    ray_vec : np.ndarray
        Direction of the ray.

    triangle : np.ndarray
        ``3 x 3`` numpy array containing the three vertices of a
        triangle.

    Returns
    -------
    bool
        ``True`` when there is an intersection.

    tuple
        Length three tuple containing the distance ``t``, and the
        intersection in unit triangle ``u``, ``v`` coordinates.  When
        there is no intersection, these values will be:
        ``[np.nan, np.nan, np.nan]``

    """
    def pass_fnc():
        pass
    # define a null intersection
    null_inter = jnp.array([jnp.nan, jnp.nan, jnp.nan])

    # break down triangle into the individual points
    v1, v2, v3 = vertexes
    
    

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = jnp.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if jnp.absolute(det) < eps:  # no intersection
        return False, null_inter
    inv_det = 1.0 / det
    tvec = ray_start - v1
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 or u > 1.0:  # if not intersection
        return False, null_inter

    qvec = jnp.cross(tvec, edge1)
    v = ray_vec.dot(qvec) * inv_det
    
    if v < 0.0 or u + v > 1.0:  # if not intersection
        return False, null_inter

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, null_inter

    # print('True')
    # print(f'ray_vec: {ray_vec}')
    return True, jnp.array([t, u, v])

# @partial(jax.jit, static_argnames=['eps'])
# def _intersect(vertexes, ray_start, ray_vec, eps=0.000001):
#     null_inter = jnp.array([jnp.nan, jnp.nan, jnp.nan])
#     v1, v2, v3 = vertexes

#     edge1 = v2 - v1
#     edge2 = v3 - v1
#     pvec = jnp.cross(ray_vec, edge2)
#     det = jnp.dot(edge1, pvec)

#     def det_small_fn(_):
#         return False, null_inter

#     def det_not_small_fn(_):
#         inv_det = 1.0 / det
#         tvec = ray_start - v1
#         u = jnp.dot(tvec, pvec) * inv_det

#         def u_out_of_range_fn(_):
#             return False, null_inter

#         def u_in_range_fn(_):
#             qvec = jnp.cross(tvec, edge1)
#             v = jnp.dot(ray_vec, qvec) * inv_det

#             def v_out_of_range_fn(_):
#                 return False, null_inter

#             def v_in_range_fn(_):
#                 t = jnp.dot(edge2, qvec) * inv_det

#                 def t_small_fn(_):
#                     return False, null_inter

#                 def t_not_small_fn(_):
#                     return True, jnp.array([t, u, v])

#                 return jax.lax.cond(t < eps, t_small_fn, t_not_small_fn, None)

#             return jax.lax.cond(v < 0.0 or u + v > 1.0, v_out_of_range_fn, v_in_range_fn, None)

#         return jax.lax.cond(u < 0.0 or u > 1.0, u_out_of_range_fn, u_in_range_fn, None)

#     return jax.lax.cond(jnp.abs(det) < eps, det_small_fn, det_not_small_fn, None)

