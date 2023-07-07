from Shapes import Triangle


class Ray():

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def set_collision(self, triangle: Triangle):
        self.set_collision = triangle
