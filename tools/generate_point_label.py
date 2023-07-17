import os
import os.path as osp
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
}
filter_lidarseg_classes = tuple(NameMapping.keys())

class_names = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
)
class_idx = dict()
for i, name in enumerate(class_names):
    class_idx[name] = i

camera_names = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
)

def generate(dataroot, save_dir):
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    filter_lidarseg_labels = []
    for class_name in filter_lidarseg_classes:
        filter_lidarseg_labels.append(nusc.lidarseg_name2idx_mapping[class_name])

    for i, sample in tqdm(enumerate(nusc.sample)):
        sample_data = sample['data']
        pointsensor_token = sample_data['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        lidarseg_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', pointsensor_token)['filename'])
        points_seg_raw = np.fromfile(lidarseg_filename, dtype=np.uint8).astype(np.int16)
        assert len(points_seg_raw)==pc.nbr_points(), "lidarseg size not equal to lidar points"

        # map the label of points, -1 means background
        points_seg = np.zeros(points_seg_raw.shape, dtype=np.int16) - 1
        for lidarseg_label in filter_lidarseg_labels:
            points_seg[points_seg_raw == lidarseg_label] = \
                class_idx[NameMapping[nusc.lidarseg_idx2name_mapping[lidarseg_label]]]

        # lidar to ego
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        for camera_name in camera_names:
            camera_token = sample_data[camera_name]
            cam = nusc.get('sample_data', camera_token)
            cam_filename = cam['filename']

            # ego to camera
            pc_tmp = copy.deepcopy(pc)
            points_seg_tmp = copy.deepcopy(points_seg)
            poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            pc_tmp.translate(-np.array(poserecord['translation']))
            pc_tmp.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

            cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pc_tmp.translate(-np.array(cs_record['translation']))
            pc_tmp.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            points_depth = pc_tmp.points[2, :]      
            points = view_points(pc_tmp.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
            mask = np.ones(points_depth.shape[0], dtype=bool)
            mask = np.logical_and(mask, points_depth > 1.0)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < 1600 - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < 900 - 1)
            points = points[:, mask]
            points_seg_tmp = points_seg_tmp[mask]
            points_depth = points_depth[mask]
            points_label = np.concatenate([points[:2], points_depth[np.newaxis,:], 
                                           points_seg_tmp[np.newaxis,:]], axis=0)
            label_save_path = cam_filename.replace('samples', save_dir).replace('.jpg', '.npy')
            label_save_path = osp.join(nusc.dataroot, label_save_path)

            if not osp.exists(osp.dirname(label_save_path)):
                os.makedirs(osp.dirname(label_save_path))
            np.save(label_save_path, points_label)

if __name__=='__main__':
    dataroot = 'data/nuscenes/'
    save_dir = 'samples_point_label'
    generate(dataroot, save_dir)