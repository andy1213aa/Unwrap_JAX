import jax.numpy as jnp
from Ray import Ray
from Camera import Camera
class InverseRender():
    
    '''
    Class responsible for inverse rendering the vertex color.
    '''
    
    def __init__(self, camera:Camera):
        self.camera = camera
       
        
    def ray_color(self, ray:Ray):
        pass
    
    def ray_inverse_trace(self, img, verts, ):
        
        for v in verts:
            
            ray = Ray(v, )