import cv2
import matplotlib.pyplot as plt
import numpy as np
from Camera import Camera
from Render import Renderer
from pytorch3d.io import load_obj
from Shapes import Triangle
from KDtree import KDTree
from Vis3D import Vis_tracing_3D
import time
from Ray import Ray
import utlis
import torch


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
    view = '400002'
    idx = '000220'
    texturesize = (1024, 1024, 3)
    render_type = 'feature_marking'  #'feature_marking'  #render_texel
    # obj_pth = f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.obj'
    img2D_pth = f'/home/aaron/Desktop/multiface/{subject}_GHS/images/{facial}/{view}/{idx}.png'
    obj_pth = 'MeshroomCache/Texturing/93c9631dda87041b9fef6beebf2734ecc11d0240/texturedMesh.obj'
    # img2D_pth = '/home/aaron/Downloads/me/test/frame_0001.png'
    krt_pth = '/home/aaron/Desktop/multiface/6674443_GHS/KRT'
    '''
    Get R t from Meshroom
    '''
    R_o = [
        "0.95157216271871836", "0.16986488511456146", "-0.256234931154669",
        "-0.16874388799072634", "0.98530298599920207", "0.026524065428010543",
        "0.25697455011069548", "0.017998516218650118", "0.96625055446780095"
    ]

    C_o = [
        "0.48676346278241278", "-0.022171128484136042", "0.08938904329579897"
    ]

    R = np.array([float(i) for i in R_o]).reshape(3, 3)
    C = np.array([float(i) for i in C_o]).reshape(3, 1)
    t = (-1 * R @ C).reshape(3, )

    print(f'R: {R}')
    print(f't: {t}')
    '''
    2D IMAGE
    '''
    img = cv2.imread(img2D_pth)
    cv2.imwrite('2dimage.png', img)
    '''
    CAMERA
    '''
    camera_params = get_KRT(krt_pth)
    camera = Camera(
        origin=np.array([0., 0., 0.]),
        width=1334,
        height=2048,
        K=camera_params[view]['K'],
        # R=camera_params[view]['R'],
        # t=camera_params[view]['t'],
        R=R,
        t=t,
        focal=camera_params[view]['focal'],
        princpt=camera_params[view]['princpt'],
        img=img,
        pixel_size_mm=0.00345)
    '''
    OBJ
    '''

    verts, faces, aux = load_obj(obj_pth)
    verts = verts.numpy()
    verts = (camera.R @ verts.T).T + camera.t

    verts[:, 2] *= -1
    print(verts)
    scene_objects = [
        Triangle(i, verts[vtidx]) for i, vtidx in enumerate(faces.verts_idx)
    ]

    used_vtx_idx = np.unique(faces.verts_idx.numpy().flatten())
    vis3d = Vis_tracing_3D(verts, used_vtx_idx)

    start = time.time()
    print('Building kdtree...')
    kd_tree = KDTree(scene_objects, vis3d)
    print('Building kdtree done.')
    print(f'kdtree time: {time.time() - start:.5f}s')
    '''
    RAY
    '''
    if render_type == 'feature_marking':

        faceDetection = utlis.FaceMesh()
        feature = faceDetection.detect(np.expand_dims(img, 0))[0]
        pts1 = np.zeros((feature.shape[0], 3), dtype=np.float32)

        feature_mm = np.apply_along_axis(camera.pixel2xy, axis=1, arr=feature)

        focal_mm_col = np.full((feature.shape[0], 1), camera.focal_mm[0])

        pts2 = np.append(feature_mm, focal_mm_col, axis=1)

    elif render_type == 'render_texel':
        pts1 = verts
        pts2 = camera.origin

    Rays = utlis.create_rays(pts1=pts1, pts2=pts2)

    renderer = Renderer(camera, kd_tree, Rays)
    res = renderer.feature_marking()
    # vertex_color = renderer.render_texel()
    # '''
    # Draw
    # '''

    # uv_texture = np.zeros(texturesize).astype(np.uint8)
    # vert2uv = {}

    # for i, idxes in enumerate(faces.verts_idx.numpy()):
    #     for j, idx in enumerate(idxes):
    #         if idx not in vert2uv:
    #             vert2uv[idx] = (
    #                 aux.verts_uvs.numpy()[faces.textures_idx[i][j]] *
    #                 1023).astype(np.int32)

    # for i, vtx_c in enumerate(vertex_color):
    #     if i in vert2uv:
    #         uv = vert2uv[i]
    #         uv_texture[uv[0], uv[1]] = vtx_c
    #         cv2.circle(uv_texture, (uv[0], uv[1]), 10, tuple(vtx_c), -1)

    # cv2.imwrite('uv_test.png', uv_texture)
    '''
    VIS
    '''

    # # 创建示例数据
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # 创建3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, tri in enumerate(res):
        middle = tri.vertices[0] * 0.3 + tri.vertices[1] * 0.4 + tri.vertices[
            2] * 0.3
        ax.scatter(middle[0], middle[1], middle[2], s=10, c='r')

    # # 绘制3D点
    # ax.scatter(x, y, z, s=[1] * 7306, c=vertex_color / 255.)

    # ax.scatter(x, y, z, s=[1] * verts.shape[0], c=[(0, 0, 0)] * verts.shape[0])
    # ax.scatter(0, 0, 0, s=[100], c='g')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()
    '''
    '''


if __name__ == '__main__':

    main()