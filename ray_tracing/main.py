import cv2
import matplotlib.pyplot as plt
import numpy as np
from .Camera import Camera
from .Render import Renderer
from pytorch3d.io import load_obj
from .Shapes import Triangle
from .KDtree import KDTree
from .Vis3D import Vis_tracing_3D
import time
from .utlis import *


def main(width,
         height,
         R,
         t,
         focal,
         princpt,
         pixel_size_mm,
         obj_pth,
         img_pth,
         render_type,
         visualize=True):
    # render_type = 'feature_marking'  #'feature_marking'  #render_texel
    img = cv2.imread(img_pth)
    '''
    CAMERA
    '''
    camera = Camera(
        origin=np.array([0., 0., 0.]),
        width=width,
        height=height,
        R=R,
        t=t,
        focal=np.array([focal] * 2),
        princpt=princpt,
        pixel_size_mm=pixel_size_mm,
    )
    '''
    OBJ
    '''

    verts, faces, aux = load_obj(obj_pth)
    verts = verts.numpy()
    verts = (camera.R @ verts.T).T + camera.t
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

        # feature detection
        faceDetection = FaceMesh()
        feature = faceDetection.detect(np.expand_dims(img, 0))[0]

        # pts1
        pts1 = np.zeros((feature.shape[0], 3), dtype=np.float32)

        # pts2
        feature_mm = np.apply_along_axis(camera.pixel2xy, axis=1, arr=feature)
        focal_mm_col = np.full((feature.shape[0], 1), camera.focal_mm[0])
        pts2 = np.append(feature_mm, focal_mm_col, axis=1)

    elif render_type == 'render_texel':
        pts1 = verts
        pts2 = camera.origin

    Rays = create_rays(pts1=pts1, pts2=pts2)
    renderer = Renderer(camera, kd_tree, Rays)
    res, flag = renderer.feature_marking()
    '''
    VIS
    '''
    if visualize:
        clip_num = 50

        x = verts[:, 0]
        x_clip = [x[i] for i in range(0, len(x), clip_num)]
        y = verts[:, 1]
        y_clip = [y[i] for i in range(0, len(y), clip_num)]
        z = verts[:, 2]
        z_clip = [z[i] for i in range(0, len(z), clip_num)]

        # 创建3D图形对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, tri in enumerate(res):
            middle = tri.vertices[0] * 0.3 + tri.vertices[
                1] * 0.4 + tri.vertices[2] * 0.3
            ax.scatter(middle[0], middle[1], middle[2], s=10, c='r')

        # # 绘制3D点
        # ax.scatter(x, y, z, s=[1] * 7306, c=vertex_color / 255.)

        ax.scatter(x_clip,
                   y_clip,
                   z_clip,
                   s=[1] * len(x_clip),
                   c=[(0, 0, 0)] * len(x_clip))
        ax.scatter(0, 0, 0, s=[100], c='g')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 显示图形
        plt.show()

    return res, flag