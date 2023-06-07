
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.io import load_obj
import numpy as np

verts, faces, aux = load_obj('../test_data/013665.obj')
verts = verts.numpy()
used_vtx_idx = np.unique(faces.verts_idx.numpy().flatten())
used_verts = verts[used_vtx_idx]
print(used_verts.shape)
color = ['r']*7306
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
ax.scatter(x, y, z, s=[1]*7306, c=color)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()