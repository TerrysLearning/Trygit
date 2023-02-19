# from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

def is_within_bb(points, bb_min, bb_max):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)

def getitem(scene, labels, heuristic, noise):
    ret = {}
    # 把point转换成voxel
    input_coords = scene["positions"] - min(0, np.min(scene["positions"])) # Translate scene to avoid negative coords
    voxel_size = 0.02
    input_coords = input_coords / voxel_size  # Scale to voxel size
    vox_coords = np.round(input_coords) # from here on our voxels coordinates represent the center location of the space they discretize
    ret['vox_coords'], vox2point = np.unique(vox_coords, axis=0, return_inverse=True)  
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(input_coords)
    point2vox = nbrs.kneighbors(ret['vox_coords'], return_distance=False)
    point2vox = point2vox.reshape(-1)  # (num_voxels) # point2vox maps an array organized as scene points to an array organized as vox_coords: num_points->num_voxels
    # input feature 是 color + normal
    input_feats = [scene["colors"]]
    input_normals = scene["normals"]
    input_feats.append(input_normals)
    input_feats = np.concatenate(input_feats, 1) 
    #### Voxelize the input to the network (scene)
    ret['vox_segments'] = scene['segments'][point2vox]  # (num_voxels)
    ret['vox_features'] = input_feats[point2vox]  # (num_voxels, feature_dim)
    ret['scene'] = scene
    ret['vox_world_coords'] = ret['vox_coords'] * voxel_size + min(0, np.min(scene["positions"]))
    ret['vox2point'] = vox2point
    ret['point2vox'] = point2vox
    unique_vox_segments = None # initialization
    # define GT per segment
    unique_vox_segments, seg2vox = np.unique(ret['vox_segments'], return_inverse = True) # (unique_vox_segments), (num_voxels)
    seg2point = seg2vox[vox2point]
    segment_middle = np.zeros((unique_vox_segments.shape[0], 3))
    segment_middle.fill(np.nan)
    for i, segment in enumerate(unique_vox_segments):
        segment_coords = ret['vox_world_coords'][segment == ret['vox_segments']]
        segment_middle[i] = np.mean(segment_coords, axis=0)
    assert ~ np.any(np.isnan(segment_middle))
    ret['input_location'] = segment_middle
    ret['seg2point'] = seg2point
    ret['seg2vox'] = seg2vox
    ret['pred2point'] = seg2point
    ret['labels'] = labels
    # if unique_vox_segments is None:
    #     unique_vox_segments = np.unique(ret['vox_segments'])  # (unique_vox_segments), (num_voxels)
    bbs_supervision(ret, labels, scene, unique_vox_segments, heuristic, noise)
    return ret

def bbs_supervision(ret, labels, scene, unique_segs, heuristic, noise):
    inst_per_point, inst_per_seg = approx_association(labels, scene, unique_segs, heuristic, noise)
    ret['pseudo_inst'] = inst_per_point, inst_per_seg # just for visualization / analysis purposes
    instances = inst_per_seg  # len=oversegs
    segments_instances = labels['seg2inst'][unique_segs]
    gt_full_sem = labels['per_instance_semantics'][segments_instances] # full supervision semantics
    gt_unlabeled = gt_full_sem == 0 # scannet has missing annotations, don't supervise on those
    fg_instances = instances > -1 # exclude bg and unknown
    ret['fg_instances'] = fg_instances  # bool list 
    # ------------------ GT INSTANCES
    gt_bb_bounds = np.zeros((len(fg_instances), 3))
    gt_bb_bounds[fg_instances] = labels['per_instance_bb_bounds'][instances[fg_instances]] 
    ret['gt_bb_bounds'] = gt_bb_bounds
    gt_bb_centers = np.zeros((len(fg_instances), 3))
    gt_bb_centers[fg_instances] = labels['per_instance_bb_centers'][instances[fg_instances]]
    ret['gt_bb_offsets'] = gt_bb_centers - (ret['input_location'] * fg_instances[:,None] + 0)
    ret['gt_bb_centers'] = gt_bb_centers
    # ------------------ GT SEMANTICS
    # zero label is corresponds to 'unlabeled' and is ignored in loss computation
    gt_semantics = np.zeros(len(fg_instances), dtype=np.int)
    # we use semantics of instances only where we have instances
    gt_semantics[fg_instances] = labels['per_instance_semantics'][instances[fg_instances]]  
    # for the "-1" (pseudo) background class, we predict, "2" (floor label in original ScanNet)
    # gt_semantics[instances == -1] = 2 
    # gt_semantics[instances == -2] = 0 this is implicit by having a starting value of 0
    gt_semantics[gt_unlabeled] = 0 # scannet has missing annotations, keep those as 'unlabeled' class
    ret['gt_semantics'] = gt_semantics
    error(gt_semantics, gt_full_sem, fg_instances, gt_unlabeled)


def approx_association(labels, scene, unique_segs, heuristic, noise):
    # ---------------- FIND APPROX. BOXES to POINT ASSOCIATIONS --------------------------
    # for each point we find its approximate instance id # remove bounding boxes from walls / floor / ceiling
    semantics = labels['per_instance_semantics']
    scene_fg = (semantics > 2) & (semantics != 22)  # also excludes unlabeled
    # get bounding boxes
    centers = labels['per_instance_bb_centers'][scene_fg]
    bounds = labels['per_instance_bb_bounds'][scene_fg] + 0.005 # 0.005
    min_corner = centers - bounds
    max_corner = centers + bounds
    instance_ids = labels['unique_instances'][scene_fg]  # id of each bounding box
    # compute per bb, what points are within it # put into matrix: bbs x points
    if noise>0:
        noisy_boxes = noise
        rng = np.random.default_rng(seed=3000) 
        min_corner += rng.normal(loc=0, scale=noisy_boxes/2, size= min_corner.shape) # scale is std dev
        max_corner += rng.normal(loc=0, scale=noisy_boxes/2, size= max_corner.shape) # scale is std dev
        
    bb_occupancy = is_within_bb(scene['positions'], min_corner[:, None], max_corner[:, None])
    # stores for each point: list of what BBs indices it is contained in
    activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
    num_BBs_per_point = bb_occupancy.sum(axis=0)
    bb_volume = np.prod(2 * bounds, axis=1)

    if heuristic == 'mode':
        inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
        for i, activ in enumerate(activations_per_point):
            if num_BBs_per_point[i] == 1:  # if point is active in one row (bb)
                bb_idx = activ[0, 0]
                inst_per_point[i] = instance_ids[bb_idx]  # add the row idx (bb idx)
            if num_BBs_per_point[i] > 1:  # multiple BBs associated, label unknown
                box_ids = activ.reshape(-1)
                smallest_box_id = np.argmin(bb_volume[box_ids])
                inst_id = instance_ids[box_ids[smallest_box_id]]
                inst_per_point[i] = inst_id # smallest bb

        inst_per_point_maj_pooled = np.ones(len(scene['positions']), dtype=np.int) * -2
        inst_per_seg_maj_pooled = np.ones(len(unique_segs), dtype=np.int) * -2
        for i, seg_id in enumerate(unique_segs):
            seg_mask = seg_id == scene['segments']
            ins_id = stats.mode(inst_per_point[seg_mask], None)[0][0]
            inst_per_point_maj_pooled[seg_mask] = ins_id
            inst_per_seg_maj_pooled[i] = ins_id

        return inst_per_point_maj_pooled, inst_per_seg_maj_pooled

    else:
        inst_per_point_pooled = np.ones(len(scene['positions']), dtype=np.int) * -2
        inst_per_seg_pooled = np.ones(len(unique_segs), dtype=np.int) * -2
        for i, seg_id in enumerate(unique_segs):
            seg_mask = seg_id == scene['segments']
            num_BBs_on_seg = num_BBs_per_point[seg_mask]
            min_BBs_on_seg = num_BBs_on_seg.min()   
            if min_BBs_on_seg == 1:
                seg_idx = np.where(num_BBs_on_seg == 1)[0][0]  # get index of point within segment
                seg_idx2point_idx = np.where(seg_mask)[0]
                point_idx = seg_idx2point_idx[seg_idx]  # get index of point within scene
                bb_idx = activations_per_point[point_idx][0, 0]  # get BB of that point
                inst_per_point_pooled[seg_mask] = instance_ids[bb_idx]  # set the same BB for all seg points
                inst_per_seg_pooled[i] = instance_ids[bb_idx]  # set the same BB for all seg points
            if min_BBs_on_seg == 0: # no BB, this is background. We set the -1
                inst_per_point_pooled[seg_mask] = -1
                inst_per_seg_pooled[i] = -1

        # smallest_bb_heuristic:
        undecided_segs = np.where(inst_per_seg_pooled == -2)[0]
        for undecided_seg in undecided_segs:
            seg_id = unique_segs[undecided_seg]
            seg_mask = seg_id == scene['segments']
            num_BBs_on_seg = num_BBs_per_point[seg_mask]
            seg_point = num_BBs_on_seg.argmin()
            point_idx = np.where(seg_mask)[0][seg_point]
            box_ids = activations_per_point[point_idx].reshape(-1)
            smallest_box_id = np.argmin(bb_volume[box_ids])
            inst_id = instance_ids[box_ids[smallest_box_id]]  # 这里
            inst_per_seg_pooled[undecided_seg] = inst_id
            inst_per_point_pooled[seg_mask] = inst_id

        return inst_per_point_pooled, inst_per_seg_pooled # return per point and per segment annotation


def error(gt_semantics, gt_full_sem, fg_instances, gt_unlabeled):
    num_1 = 0 # miss labeled background into fg_instance
    num_2 = 0 # miss labeled fg_instance into background
    num_3 = 0 # miss labeled fg_instance into another
    segs_n = len(gt_semantics)
    unlab_n = len(np.where(gt_unlabeled==True)[0])
    lab_n = segs_n - unlab_n
    print('total labeled', lab_n)
    for i in range(segs_n):
        if fg_instances[i] and (not is_foreground(gt_full_sem[i])) and (not gt_unlabeled[i]):
            num_1 += 1
        elif (not fg_instances[i]) and is_foreground(gt_full_sem[i]) and (not gt_unlabeled[i]):
            num_2 += 1
        elif fg_instances[i] and is_foreground(gt_full_sem[i]) and (not gt_unlabeled[i]): 
            if gt_semantics[i] != gt_full_sem[i]:
                num_3 += 1
    print('miss labels:', num_1, num_2, num_3)
    print('correct label rate', (lab_n-num_1-num_2-num_3)/lab_n)

def is_foreground(sem):
    return (sem > 2) & (sem != 22)