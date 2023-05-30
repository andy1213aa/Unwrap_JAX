import jax.numpy as jnp


class Camera:
    '''
    A class for keeping track of camera parameters.
    '''

    def __init__(
        self,
        origin,
        width: int,
        height: int,
        K,
        R,
        t,
        focal: float,
        princpt: float,
    ):
        self.origin = origin
        self.width = width
        self.height = height
        self.K = K
        self.R = R
        self.t = t
        self.focal = focal
        self.princpt = princpt
        
