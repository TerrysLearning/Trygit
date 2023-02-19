"""Prepare scannet data. Read the ground truth instance and semantic labels, and compute centers."""
import os
import json
import csv
import numpy as np
import open3d as o3d
from scipy.stats import rankdata
import math
# import augmentation

def rand_rotation(n):
    # generate a rotation matrix Rr
    Rr = np.eye(4)
    rotations = np.array([0, 0.5* np.pi, np.pi, 1.5 * np.pi])
    # angle = rotations[np.random.randint(0,4)]
    angle = rotations[n]
    Rr[0][0] = np.cos(angle)
    Rr[1][0] = np.sin(angle)
    Rr[0][1] = -np.sin(angle)
    Rr[1][1] = np.cos(angle)
    return Rr

def hom_transfer_all(ps, Rt):
    ps_ = []
    for p in ps:
        p_ = np.concatenate((p, np.array([1])))
        p_ = Rt@p_
        ps_.append(p_[:3])
    return np.array(ps_)


def read_scene(path_ply, path_txt, align=True, argumentation=False):
    """Read the scene .ply.

    :return
        positions: 3D-float position of each vertex/point
        normals: 3D-float normal of each vertex/point (as computed by open3d)
        colors: 3D-float color of each vertex/point [0..1]
    """
    mesh = o3d.io.read_triangle_mesh(path_ply)

    if align:
        with open(path_txt) as f:
            lines = f.readlines()
        axis_alignment = ''
        for line in lines:
            if line.startswith('axisAlignment'):
                axis_alignment = line
                break
        if axis_alignment == '':
            raise ValueError('No axis alignment found!')
        Rt = np.array([float(v) for v in axis_alignment.split('=')[1].strip().split(' ')]).reshape([4, 4])
        mesh.transform(Rt)

    # Apply geometric augmentation
    scaling_aug = [1.0, 0.8, 1.2]
    flipping_aug = 0.5  # x axis flip, vertex x go to negative, norm x no change, others go to negative
    rotation_90_aug = True #  
    apply_hue_aug = True # times vertex 

    positions = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)

    # if argumentation:
    #     
    # Rr = rand_rotation(1)
    # print(Rr)
    # positions = hom_transfer_all(positions, Rr)
    # normals = hom_transfer_all(normals, Rr)

    # print(positions[0:10])
    # print(normals[0:10])
    # print(colors[0:10])
    # print(triangles[0:10])

    return positions, normals, colors, triangles

def read_labels(label_map_file, path_aggregation, per_point_segment_ids):

    # Create label map, i.e. map from label_name to label_id
    label_map = {}
    with open(label_map_file, 'r') as f:
        lines = csv.reader(f, delimiter='\t')
        cnt = 0
        for line in lines:
            if cnt > 0:
                if len(line[4]) > 0:
                    label_map[line[1]] = line[4]
                else:
                    label_map[line[1]] = '0'
            cnt += 1
    # print(label_map, 'hahahahhahha')
    # Read semantic labels
    with open(path_aggregation) as f:
        aggregation_data = json.load(f)

    # semantics and instances are overwritten with non-zero values in the following
    # if 0 values are encountered this means the point had no annotation - i.e. zero is our default here
    per_point_semantic_labels = np.zeros((len(per_point_segment_ids)), dtype='int32')
    per_point_instance_labels = np.zeros((len(per_point_segment_ids)), dtype='int32')

    summ = 0 
    for instance_id, instance in enumerate(aggregation_data["segGroups"]):  
        semantic_string = instance["label"]
        summ += len(instance['segments'])
        for segment in instance["segments"]:
            ind = per_point_segment_ids == int(segment)
            if semantic_string in label_map:
                semantic_id = label_map[semantic_string]
            else:
                semantic_id = '-'  # 这里没有跑进过
            per_point_semantic_labels[ind] = int(semantic_id)
            per_point_instance_labels[ind] = instance_id + 1
    # print(len(np.where(per_point_semantic_labels!=0)[0])-len(per_point_semantic_labels))

    # There are some buggy scenes (like e.g. scene0217_00) that double define instances
    # here I handle this bug:
    unique_instance_ids = np.unique(per_point_instance_labels)
    if not np.all(unique_instance_ids == range(len(unique_instance_ids))):
        per_point_instance_labels = rankdata(per_point_instance_labels, method='dense') - 1  # (num_voxels_in_batch)

    # create vectorized seg_id to instance mapping
    unique_segments_ids = np.unique(per_point_segment_ids)
    seg2inst = np.zeros(np.max(unique_segments_ids) + 1, dtype='int32')
    seg2inst.fill(np.inf)
    for seg_id in unique_segments_ids:
        seg_mask = per_point_segment_ids == seg_id
        assert len(np.unique(per_point_instance_labels[seg_mask])) == 1
        inst_id = per_point_instance_labels[seg_mask][0]
        seg2inst[seg_id] = inst_id

    return per_point_semantic_labels, per_point_instance_labels, seg2inst

def compute_avg_centers(positions, instance_labels):
    per_point_centers = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_offsets = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_center_distances = np.zeros((instance_labels.shape[0], 1), dtype='float32')

    for instance_id in set(instance_labels):
        instance_mask = (instance_id == instance_labels)

        # compute AVG centers
        instance_center = np.mean(positions[instance_mask], axis=0)
        per_point_centers[instance_mask] = instance_center
        per_point_offsets[instance_mask] = per_point_centers[instance_mask] - positions[instance_mask]
        per_point_center_distances = np.linalg.norm(per_point_offsets, axis=1)

    return per_point_centers, per_point_center_distances

def compute_bounding_box(positions, instance_labels, semantic_labels):
    per_point_bb_centers = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_offsets = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_bounds = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_center_distances = np.zeros((instance_labels.shape[0], 1), dtype='float32')
    per_point_bb_radius = np.zeros((instance_labels.shape[0], 1), dtype='float32')

    instances = np.unique(instance_labels)
    per_instance_semantics = np.zeros((len(instances)),  dtype='int32')
    per_instance_bb_centers = np.zeros((len(instances), 3), dtype='float32')
    per_instance_bb_bounds = np.zeros((len(instances), 3), dtype='float32')
    per_instance_bb_radius = np.zeros((len(instances)), dtype='float32')

    for i, instance_id in enumerate(instances):
        instance_mask = (instance_id == instance_labels)
        instance_points = positions[instance_mask]
        per_instance_semantics[i] = semantic_labels[instance_mask][0]

        # bb center
        max_bounds = np.max(instance_points, axis=0)
        min_bounds = np.min(instance_points, axis=0)
        bb_center = (min_bounds + max_bounds) / 2
        per_point_bb_centers[instance_mask] = bb_center
        per_instance_bb_centers[i] = bb_center

        # bb bounds
        bb_bounds = max_bounds - bb_center
        per_point_bb_bounds[instance_mask] = bb_bounds
        per_instance_bb_bounds[i] = bb_bounds

        # bb center offsets
        offsets = bb_center - instance_points
        per_point_bb_offsets[instance_mask] =  offsets

        # bb center distances
        # import ipdb; ipdb.set_trace()
        bb_center_distances = np.linalg.norm(offsets, axis=1)
        per_point_bb_center_distances[instance_mask] = bb_center_distances.reshape((-1,1))

        # bb radius
        radius = np.max(bb_center_distances).reshape((-1,1))
        per_point_bb_radius[instance_mask] = radius
        per_instance_bb_radius[i] = radius

    return per_point_bb_centers, per_point_bb_offsets, per_point_bb_bounds, \
           per_point_bb_center_distances, per_point_bb_radius, \
           instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius

def process_scene(scene_name, a):
    """Process scene: extracts ground truth labels (instance and semantics) and computes centers

    :return
        scene: dictionary containing
            positions: 3D-float position of each vertex/point
            normals: 3D-float normal of each vertex/point (as computed by open3d)
            colors: 3D-float color of each vertex/point [0..1SEMANTIC_VALID_CLASS_IDS]
        labels:  dictionary containing
            semantic_labels: N x 1 int32
            instance_labels: N x 1 int32
            centers: N x 3 float32
            center_distances: N x 1 float32
    """
    # Setup pathes to all necessary files
    data_path = 'scans'
    path_segmention = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.0.010000.segs.json')
    path_txt = os.path.join(data_path, scene_name, f'{scene_name}.txt')
    path_ply = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.ply')
    path_aggregation = os.path.join(data_path, scene_name, f'{scene_name}.aggregation.json')
    #path_aggregation = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean.aggregation.json')
    label_map_file = os.path.join(data_path, 'scannetv2-labels.combined.tsv')

    # ----------------- INPUT SCENE ----------------------------#
    # Read point clouds, extract semantic & instance labels, compute centers
    positions, normals, colors, triangles = read_scene(path_ply, path_txt, align=True, argumentation=a)
    with open(path_segmention) as f:
        per_point_segment_ids = json.load(f)
    per_point_segment_ids = np.asarray(per_point_segment_ids["segIndices"], dtype='int32')
    scene = {'name': scene_name, 'positions': positions, 'normals': normals, 'colors': colors, 'segments': per_point_segment_ids, 'triangles': triangles}
    # ----------------- GT LABELS  ----------------------------#
    semantic_labels, instance_labels, seg2inst = read_labels(label_map_file, path_aggregation, per_point_segment_ids)

    centers, center_distances = compute_avg_centers(positions, instance_labels)

    bb_centers, bb_offsets, bb_bounds, bb_center_distances, bb_radius, \
    unique_instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius \
    = compute_bounding_box(positions, instance_labels, semantic_labels)

    # make sure the unique instance ids can be used as array indices for 'per_instance_XX'
    assert np.all(unique_instances == range(len(unique_instances)))
    
    labels = {'semantics': semantic_labels, 'instances': instance_labels,
              'centers': centers, 'center_distances': center_distances,
              'bb_centers':bb_centers, 'bb_offsets':bb_offsets, 'bb_bounds':bb_bounds, 'seg2inst': seg2inst,
              'bb_center_distances': bb_center_distances, 'bb_radius':bb_radius,
              'unique_instances':unique_instances, 'per_instance_semantics':per_instance_semantics,
              'per_instance_bb_centers':per_instance_bb_centers, 'per_instance_bb_bounds': per_instance_bb_bounds,
              'per_instance_bb_radius': per_instance_bb_radius }

    return scene, labels



