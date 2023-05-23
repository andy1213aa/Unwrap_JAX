import jax



class Ray:
    '''
    A class for ray object parameterized as
        P(t) = A + tb
    where A is the ray origin, b is the ray direction, and
    t is the length of the vector.
    '''

    def __init__(self, ray_origin, ray_direction):
        self.ray_origin = ray_origin
        self.ray_direction = ray_direction

    def __str__(self):
        print(
            f"P(t) = ({self.ray_origin[0]:.3}, {self.ray_origin[1]:.3}, {self.ray_origin[2]:.3}) + t({self.ray_direction[0]:.3}, {self.ray_direction[1]:.3}, {self.ray_direction[2]:.3})"
        )

    def at(self, t):
        '''
        Evaluates the ray at the point t.
        @param t: How long to go along the ray direction
        @return: P(t), the ray evaluated at t
        '''
        return self.ray_origin + t * self.ray_direction
