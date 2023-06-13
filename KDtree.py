from AABB import AABB
import numpy as np
import time


class KDNode:

    def __init__(self, objects, depth=0):
        self.left = None
        self.right = None
        self.aabb = None
        self.objects = objects
        self.depth = depth
        self.split_axis = depth % 3

        if len(objects) > 1:
            self.split()
        else:
            self.aabb = self.objects[0].aabb

    def split(self):

        sorted_objects = sorted(
            self.objects,
            key=lambda o: o.aabb.min_corner[self.split_axis],
        )
        # print(f'sorted_min: {sorted_objects[0].aabb.min_corner}')
        # print(f'sorted_max: {sorted_objects[-1].aabb.max_corner}')
        # print(f'sorted_objects: {len(sorted_objects)}')

        self.aabb = AABB(
            np.min([obj.aabb.min_corner for obj in sorted_objects], axis=0),
            np.max([obj.aabb.max_corner for obj in sorted_objects], axis=0),
        )

        median_idx = len(sorted_objects) // 2
        self.left = KDNode(
            sorted_objects[:median_idx],
            self.depth + 1,
        )
        self.right = KDNode(
            sorted_objects[median_idx:],
            self.depth + 1,
        )


class KDTree:

    def __init__(self, objects, vis3D):
        self.root = KDNode(objects)
        self.vis3D = vis3D

    def intersect(self, ith, ray_origin, ray_direction):
        intersections = []

        def traverse(node):

            if node.aabb.intersect(ray_origin, ray_direction):

                # ret = self.vis3D.draw_aabb_tracing(ith, ray_origin, node.aabb)
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)

                if not node.left and not node.right:
                    for obj in node.objects:

                        if obj.intersect(ray_origin, ray_direction):

                            intersections.append(obj)

        traverse(self.root)

        return intersections