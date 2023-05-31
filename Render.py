import jax.numpy as jnp
import jax
from functools import reduce
from Camera import Camera
from Ray import Ray
from Vec3 import Vec3
from Scene import Scene
import time

class Renderer:
    '''
    Class responsible for rendering an image.
    '''

    def __init__(self, camera: Camera, scene: Scene):
        self.camera = camera
        self.scene = scene

    def ray_color(self, r: Ray):
        '''
        Evaluates the color of a ray based on where it hits the background.
        '''
        unit_direction = r.ray_direction / jnp.linalg.norm(r.ray_direction)
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * jnp.array([1., 1., 1.]) + t * jnp.array(
            [0.5, 0.7, 1.0])

    def render(self):
        image = jnp.zeros((self.camera.height, self.camera.width, 3))
        for j in range(self.camera.height - 1, -1, -1):
            for i in range(self.camera.width):
                u = float(i) / (self.camera.width - 1)
                v = float(j) / (self.camera.height - 1)
                r = Ray(
                    self.camera.origin, self.camera.lower_left_corner +
                    u * self.camera.horizontal + v * self.camera.vertical -
                    self.camera.origin)
                image = jax.ops.index_update(image, jax.ops.index[j, i, :],
                                             self.ray_color(r))
        return image

    def ray_trace(self, origin, normalized_direction):
        far = 1.0e15  # A large number, which we can never hit
        # distances = [
        #     o.ray_triangle_intersection(origin, normalized_direction)
        #     for o in self.scene.objects
        # ]

        for i, o in enumerate(self.scene.objects):
            for r in normalized_direction:
                start = time.time()
                ret = o.intersect(origin, r)
                print(f'{i}th Extime: {time.time()-start}')
        # nearest = reduce(jnp.minimum, distances)
        color = Vec3(0, 0, 0)
        # for (o, d) in zip(self.scene.objects, distances):
        #     color += o.light(
        #         origin, normalized_direction, d, self.light_position, origin,
        #         self.scene.objects) * (nearest != far) * (d == nearest)
        return color

    def render_fast(self):
        r = float(self.camera.width) / self.camera.height
        # S = (-1., 1. / r + .25, 1., -1. / r + .25)
        S = (
            -self.camera.princpt[0],
            -self.camera.princpt[1],
            self.camera.princpt[0],
            self.camera.princpt[1],
        )
        x = jnp.tile(
            jnp.linspace(S[0], S[2], self.camera.width),
            self.camera.height,
        )
        y = jnp.repeat(
            jnp.linspace(S[1], S[3], self.camera.height),
            self.camera.width,
        )

        z = jnp.repeat(jnp.array([self.camera.focal[0]]), y.shape[0])
        
        print('Canvas Border:')
        print(f'X: [{jnp.min(x)}, {jnp.max(x)}]')
        print(f'Y: [{jnp.min(y)}, {jnp.max(y)}]')
        print(f'Z: [{jnp.min(z)}, {jnp.max(z)}]')

        ray = jnp.stack([x, y, z], axis=1) - self.camera.origin
        ray /= jnp.linalg.norm(ray, axis=1)[:, jnp.newaxis]

        image = self.ray_trace(
            self.camera.origin,
            ray,
        )
        # return image
