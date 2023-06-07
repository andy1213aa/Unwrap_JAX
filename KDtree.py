from AABB import AABB


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
        median_idx = len(sorted_objects) // 2

        self.left = KDNode(
            sorted_objects[:median_idx],
            self.depth + 1,
        )
        self.right = KDNode(
            sorted_objects[median_idx:],
            self.depth + 1,
        )
        self.aabb = AABB(
            self.left.aabb.min_corner,
            self.right.aabb.max_corner,
        )


class KDTree:

    def __init__(self, objects):
        self.root = KDNode(objects)

    def intersect(self, ray_origin, ray_direction):
        intersections = []
        intersections_AABB = []

        def traverse(node):
            if node.aabb.intersect(ray_origin, ray_direction):
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)
                if not node.left and not node.right:
                    intersections_AABB.append(node)
                    for obj in node.objects:
                        if obj.intersect(ray_origin, ray_direction):
                            intersections.append(obj)

        traverse(self.root)
        return intersections, intersections_AABB