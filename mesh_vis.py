
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from pytorch3d.io import load_obj
verts, faces, aux = load_obj('/home/aaron/Desktop/multiface_version/MeshroomCache/Texturing/2d439a7dd5dfcd5e43d9eaf6cfec957c7d1213de/texturedMesh.obj')

verts = verts.numpy()
used_vtx_idx = np.unique(faces.verts_idx.numpy().flatten())
used_verts = verts[used_vtx_idx]
print(verts.shape)
color = ['r']*verts.shape[0]
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
ax.scatter(x, y, z, s=[1]*verts.shape[0], c=color)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()