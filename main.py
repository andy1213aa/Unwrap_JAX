import cv2
import numpy as np
import jax
import jax.numpy as jnp
from Camera import Camera
from Render import Renderer
from pytorch3d.io import load_obj
from Shapes import triangle
from Scene import Scene


def get_KRT(camera_info_pth):
    # 定義相機參數列表
    camera_params = {}
    with open(camera_info_pth, 'r') as f:
        lines = f.readlines()
    i = 0

    while i < len(lines):
        # 讀取相機ID
        camera_id = lines[i].strip()
        i += 1
        # 讀取相機內部參數
        intrinsics = []
        for _ in range(3):
            intrinsics.append([float(x) for x in lines[i].strip().split()])
            i += 1
        intrinsics = np.array(intrinsics, dtype=np.float32).reshape((3, 3))

        #跳過一行
        i += 1

        # 讀取相機外部參數
        extrinsics = []
        for _ in range(3):
            extrinsics.append([float(x) for x in lines[i].strip().split()])
            i += 1
        extrinsics = np.array(extrinsics, dtype=np.float32).reshape((3, 4))
        R = extrinsics[:3, :3]
        t = extrinsics[:, 3]
        # 添加相機參數到dict
        camera_params[camera_id] = {
            'K': jnp.array(intrinsics),
            'R': jnp.array(R),
            't': jnp.array(t),
            'focal': jnp.array(np.array([intrinsics[0, 0], intrinsics[1, 1]])),
            'princpt': jnp.array(np.array([intrinsics[0, 2], intrinsics[1,
                                                                        2]])),
        }

        #跳過一行
        i += 1
    return camera_params


def main():

    subject = '6674443'
    facial = 'E001_Neutral_Eyes_Open'
    view = '400002'
    idx = '000220'
    '''
    2D IMAGE
    '''
    img = cv2.imread(
        f'/home/aaron/Desktop/multiface/{subject}_GHS/images/{facial}/{view}/{idx}.png'
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = jnp.array(img.astype(jnp.float32))
    '''
    OBJ
    '''
    with open('../training_data/000220.bin', 'rb') as f:
        # 读取数据
        data = f.read()
    # 解析数据
    verts = np.frombuffer(data, dtype=np.float32)
    verts = verts.reshape((7306, 3))
    verts = jnp.array(verts.astype(jnp.float32))

    verts, faces, aux = load_obj('../test_data/000220.obj')

    triangle_idx = jnp.array(faces.verts_idx)
    scene_objects = [triangle(vtxes) for vtxes in triangle_idx]

    scene = Scene(scene_objects)
    '''
    CAMERA
    '''
    camera_params = get_KRT('/home/aaron/Desktop/multiface/6674443_GHS/KRT')
    camera = Camera(
        origin=jnp.array([0., 0., 0.]),
        width=1334,
        height=2048,
        K=camera_params[view]['K'],
        R=camera_params[view]['R'],
        t=camera_params[view]['t'],
        focal=camera_params[view]['focal'],
        princpt=camera_params[view]['princpt'],
    )

    # light_position = jnp.array([[0.0, 1.0, -10.0]])
    renderer = Renderer(camera, scene)
    renderer.render_fast()
    # InRender = InverseRender(camera = camera)
    # InRender.ray_inverse_trace(img, verts)


if __name__ == '__main__':

    main()