import cv2
import numpy as np
from Camera import Camera
from Render import Renderer
from pytorch3d.io import load_obj
from Shapes import Triangle
from Scene import Scene
import torch
from KDtree import KDTree


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
            'K': np.array(intrinsics),
            'R': np.array(R),
            't': np.array(t),
            'focal': np.array(np.array([intrinsics[0, 0], intrinsics[1, 1]])),
            'princpt': np.array(np.array([intrinsics[0, 2], intrinsics[1,
                                                                       2]])),
        }

        #跳過一行
        i += 1
    return camera_params


def main():

    subject = '6674443'
    facial = 'E001_Neutral_Eyes_Open'
    view = '400004'
    idx = '000220'
    '''
    2D IMAGE
    '''
    img = cv2.imread(
        f'/home/aaron/Desktop/multiface/{subject}_GHS/images/{facial}/{view}/{idx}.png'
    )
    cv2.imwrite('2dimage.png', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img.astype(np.float32))
    '''
    CAMERA
    '''
    camera_params = get_KRT('/home/aaron/Desktop/multiface/6674443_GHS/KRT')
    camera = Camera(
        origin=np.array([0., 0., 0.]),
        width=1334,
        height=2048,
        K=camera_params[view]['K'],
        R=camera_params[view]['R'],
        t=camera_params[view]['t'],
        focal=camera_params[view]['focal'],
        princpt=camera_params[view]['princpt'],
        img=img,
    )
    '''
    OBJ
    '''
    # with open('../training_data/000220.bin', 'rb') as f:
    #     # 读取数据
    #     data = f.read()
    # # 解析数据
    # verts = np.frombuffer(data, dtype=np.float32)
    # verts = verts.reshape((7306, 3))
    # verts = np.array(verts.astype(np.float32))

    verts, faces, aux = load_obj(
        f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.obj'
    )
    verts = verts.numpy()
    print('Camera Rotation')
    print(camera.R)
    print('Camera Translation')
    print(camera.t)
    verts = (camera.R @ verts.T).T + camera.t

    print('Object Border:')
    print(f'X: [{np.min(verts[:, 0])}, {np.max(verts[:, 0])}]')
    print(f'Y: [{np.min(verts[:, 1])}, {np.max(verts[:, 1])}]')
    print(f'Z: [{np.min(verts[:, 2])}, {np.max(verts[:, 2])}]')

    scene_objects = [Triangle(verts[vtidx]) for vtidx in faces.verts_idx]
    kd_tree = KDTree(scene_objects)

    # verts = verts[np.unique(faces.verts_idx.numpy().flatten())]
    ray_dir = camera.origin - verts
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    '''
    VIS
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    used_vtx_idx = np.unique(faces.verts_idx.numpy().flatten())
    used_verts = verts[used_vtx_idx]

    color = ['r'] * 7306
    # 创建示例数据
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    for i in used_vtx_idx:
        color[i] = 'b'

    # 创建3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点
    ax.scatter(x, y, z, s=[1] * 7306, c=color)
    ax.scatter(0, 0, 0, c='r')

    #繪製平面
    u_leftup = (0 - camera.princpt[0]) / camera.focal[0]
    v_leftup = (0 - camera.princpt[1]) / camera.focal[1]

    u_rightdown = (camera.width - camera.princpt[0]) / camera.focal[0]
    v_rightdown = (camera.height - camera.princpt[1]) / camera.focal[1]

    print(f'u_leftup: {u_leftup}')
    print(f'v_leftup: {v_leftup}')

    print(f'u_rightdown: {u_rightdown}')
    print(f'v_rightdown: {v_rightdown}')

    ax.scatter(u_leftup, v_leftup, 1, c='g')
    ax.scatter(u_rightdown, v_rightdown, 1, c='g')

    # x = np.tile(np.linspace(-camera.princpt[1], camera.princpt[1], camera.width), camera.height)
    # y = np.tile(np.linspace(-camera.princpt[0], camera.princpt[0], camera.height), camera.width)
    # z = np.repeat(np.array([camera.focal[0]]), y.shape[0])
    # ax.scatter(y, x, z, c='g')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()
    '''
    '''

    renderer = Renderer(camera, kd_tree, ray_dir)
    vertex_color = renderer.render_texel()
    '''
    Draw
    '''

    uv_texture = np.zeros((1024, 1024, 3)).astype(np.uint8)

    vert2uv = {}
    for i, idxes in enumerate(faces.verts_idx.numpy()):
        for j, idx in enumerate(idxes):
            if idx not in vert2uv:
                vert2uv[idx] = (
                    aux.verts_uvs.numpy()[faces.textures_idx[i][j]] *
                    1023).astype(np.int32)

    for i, vtx_c in enumerate(vertex_color):
        if i in vert2uv:
            uv = vert2uv[i]

            # print(vtx_c.shape)

            # uv_texture[uv[0], uv[1]] = vtx_c

            cv2.circle(uv_texture, (uv[0], uv[1]), 5, tuple(vtx_c), -1)

    uv_texture = cv2.cvtColor(uv_texture, cv2.COLOR_RGB2BGR)
    cv2.imwrite('uv_test.png', np.rot90(uv_texture, k=2))
    # InRender = InverseRender(camera = camera)
    # InRender.ray_inverse_trace(img, verts)


if __name__ == '__main__':

    main()