import json
import numpy as np
import json


class View:

    def __init__(
        self,
        idx: str,
        width: int,
        height: int,
        focal: float,
        princpt: np.ndarray,
        px_size: float,
        R: np.ndarray,
        C: np.ndarray,
        img_pth: str,
    ):
        self.idx = idx
        self.width = width
        self.height = height
        self.focal = focal
        self.princpt = princpt
        self.px_size = px_size
        self.R = R
        self.C = C

        self.t = (-1 * R @ C).reshape(3, )
        self.img_pth = img_pth

    def __repr__(self) -> str:
        return f'viewId: {self.idx} path: {self.img_pth}'


def parsing_SFM_json(config):
    '''
    Loading Meshroom camera.sfm file
    '''
    newestSFMId = get_newest_folder(
        f'{config.tmp}/MeshroomCache/StructureFromMotion')

    with open(
            f'{config.tmp}/MeshroomCache/StructureFromMotion/{newestSFMId}/cameras.sfm',
            'r') as data:
        cameraSFM = json.load(data)

    intrinsics = {
        intrinsic['intrinsicId']: intrinsic
        for intrinsic in cameraSFM['intrinsics']
    }
    poses = {pose['poseId']: pose for pose in cameraSFM['poses']}

    views = []

    #每張影像都有各自的intrinsics
    #雖然大部分情況都是多張影像共用同個intrinsics
    #但寫法上然不排除不同影像有不同intrinsic的狀況
    #因此有多個ID檢索狀況
    for view in cameraSFM['views']:

        if view['poseId'] not in poses.keys():
            #有些圖片並沒有被用作Dense Reconstruction
            #沒有相機資訊，跳過
            continue
        
        principalPoint = np.array([
            float(princpt)
            for princpt in intrinsics[view['intrinsicId']]['principalPoint']
        ])

        rotation_str = poses[view['poseId']]['pose']['transform']['rotation']
        center_str = poses[view['poseId']]['pose']['transform']['center']

        rotation = np.array([float(i) for i in rotation_str]).reshape(
            (3, 3), order='F')
        center = np.array([float(i) for i in center_str]).reshape(3, 1)

        tmp = View(
            idx=view['viewId'],
            width=float(view['width']),
            height=float(view['height']),
            focal=float(intrinsics[view['intrinsicId']]['pxFocalLength']),
            princpt=principalPoint,
            px_size=-1 * float(intrinsics[view['intrinsicId']]
                               ['sensorHeight']),  # Meshroom2021版本，相機Z軸方向為負
            R=rotation,
            C=center,
            img_pth=view['path'])
        views.append(tmp)

    return views


def get_newest_folder(folder_path):
    import os

    subfolders = [
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    sorted_subfolders = sorted(
        subfolders,
        key=lambda f: os.path.getmtime(os.path.join(folder_path, f)),
        reverse=True)
    newest_folder = sorted_subfolders[0]

    return newest_folder
