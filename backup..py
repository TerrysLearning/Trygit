# def generate_point_graph(triangles, point_positions):
#     'Generate a geometric graph of each vox based on mesh connections of points'
#     graph = graph_tools.Graph(directed = False)
#     for id, _ in enumerate(point_positions):
#         connects = set() # connected point ids of this point
#         ts = [triangle for triangle in triangles if np.isin(id, triangle)] # return all triangles contains 'vid'
#         for t in ts:
#             for c_id in t:
#                 if c_id != id:
#                     connects.add(c_id)
#         for c_id in connects:
#             if not graph.has_edge(c_id, id):
#                  graph.add_edge(c_id, id) # add the edge in the graph 
#     return graph


# def generate_voxel_graph(triangles, vox_coords_unique, vox2point):
#     'Generate a geometric graph of each vox based on mesh connections of points'
#     graph = graph_tools.Graph(directed = False)
#     # from points connections into voxel connections
#     for id, _ in enumerate(vox_coords_unique):
#         point_ids = np.where(id == vox2point)[0] # take all the point indexs in the voxel
#         connects = set() # connected point ids of this voxel
#         for i in point_ids:
#             ts = [triangle for triangle in triangles if np.isin(i, triangle)] # return all triangles contains 'vid'
#             for t in ts:
#                 for p_id in t:
#                     if p_id not in point_ids:
#                         c_vox_id = vox2point[p_id]  # aggregate point id to vox id
#                         connects.add(c_vox_id)
#         for c_vox_id in connects:
#             if not graph.has_edge(c_vox_id, id):
#                  graph.add_edge(c_vox_id, id) # add the edge in the graph 
#     return graph


# pred_array = pred_offsets.detach().cpu().numpy()
#             gt_array = gt_offsets.detach().cpu().numpy()
#             error_array = np.sum(np.absolute(pred_array-gt_array), axis=1)
#             error_rank = np.argsort(-error_array)
#             wrong_ids = np.array(batch['wrong_ids'])
#             intersect = np.intersect1d(error_rank[0:len(wrong_ids)], wrong_ids, assume_unique=True)
#             print('offset wrong label percentage',len(intersect)/len(wrong_ids))

def generate_labels(scene, labels, heuristic, noise):
    vox_coords, vox_segments, unique_vox_segments, vox2point = voxel_switch(scene)
    ret = {}  
    _ , inst_per_seg = approx_association(labels, scene, unique_vox_segments, heuristic, noise)
    instances = inst_per_seg  
    segments_instances = labels['seg2inst'][unique_vox_segments]
    gt_full_sem = labels['per_instance_semantics'][segments_instances] # full supervision semantics
    gt_unlabeled = gt_full_sem == 0 # scannet has missing annotations, don't supervise on those
    fg_instances = instances > -1 # exclude bg and unknown
    ret['fg_instances'] = fg_instances  # bool list 
    # zero label is corresponds to 'unlabeled' and is ignored in loss computation
    gt_semantics = np.zeros(len(fg_instances), dtype=np.int) # we use semantics of instances only where we have instances
    gt_semantics[fg_instances] = labels['per_instance_semantics'][instances[fg_instances]]  
    # for the "-1" (pseudo) background class, we predict, "2" (floor label in original ScanNet)
    gt_semantics[instances == -1] = 2 
    # gt_semantics[instances == -2] = 0 this is implicit by having a starting value of 0
    gt_semantics[gt_unlabeled] = 0 # scannet has missing annotations, keep those as 'unlabeled' class
    ret['gt_semantics'] = gt_semantics
    error(gt_semantics, gt_full_sem, fg_instances, gt_unlabeled)
    return ret

def approx_association(labels, scene, unique_vox_segments, heuristic, noise):
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
        inst_per_seg_maj_pooled = np.ones(len(unique_vox_segments), dtype=np.int) * -2
        for i, seg_id in enumerate(unique_vox_segments):
            seg_mask = seg_id == scene['segments']
            ins_id = stats.mode(inst_per_point[seg_mask], None)[0][0]
            inst_per_point_maj_pooled[seg_mask] = ins_id
            inst_per_seg_maj_pooled[i] = ins_id

        return inst_per_point_maj_pooled, inst_per_seg_maj_pooled

    else:
        inst_per_point_pooled = np.ones(len(scene['positions']), dtype=np.int) * -2
        inst_per_seg_pooled = np.ones(len(unique_vox_segments), dtype=np.int) * -2
        for i, seg_id in enumerate(unique_vox_segments):
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
            seg_id = unique_vox_segments[undecided_seg]
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


#//////////////////////////////////////
# from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
import graph_tools 
from queue import Queue


def is_within_bb(points, bb_min, bb_max):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)


def is_foreground(sem):
    return (sem > 2) & (sem != 22)


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
            # print(i, 'is miss labeled into background' )
            num_2 += 1
        elif fg_instances[i] and is_foreground(gt_full_sem[i]) and (not gt_unlabeled[i]): 
            if gt_semantics[i] != gt_full_sem[i]:
                num_3 += 1
    print('miss labels:', num_1, num_2, num_3)
    print('correct label rate', (lab_n-num_1-num_2-num_3)/lab_n)

def box_intersect_volume(max_corner_1, min_corner_1, max_corner_2, min_corner_2):
    'input corners output intersect box max+min corner'
    max_c = np.minimum(max_corner_1, max_corner_2)
    min_c = np.maximum(min_corner_1, min_corner_2)
    if (max_c > min_c).all():
        return np.prod(max_c-min_c)
    else:
        return 0
    

def voxel_switch(scene):
    'Switch point representation to voxel representation'
    # Translate scene to avoid negative coords
    input_points = scene["positions"] - min(0, np.min(scene["positions"])) 
    # Scale to voxel size, //unit change to voxel
    voxel_size = 0.02
    input_points = input_points / voxel_size  
    vox_coords = np.round(input_points) # from here voxels coordinates represent the center location of the space they discretize
    vox_coords_unique , vox2point = np.unique(vox_coords, axis=0, return_inverse=True)  # remove same voxes of points 
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(input_points)
    point2vox = nbrs.kneighbors(vox_coords_unique, return_distance=False)
    point2vox = point2vox.reshape(-1) # point2vox maps points to scene: num_points->num_voxels
    vox_segments = scene['segments'][point2vox]  
    unique_vox_segments, _ = np.unique(vox_segments, return_inverse = True) 
    return vox_coords, vox_segments, unique_vox_segments, vox2point


def generate_point_graph(triangles, point_positions):
    'Generate a geometric graph of each vox based on mesh connections of points'
    graph = graph_tools.Graph(directed = False)
    # To reduce the computation complexity, put the triangle connections into dict first
    triangles_dict = {}
    for id, _ in enumerate(point_positions):
        triangles_dict[id] = set()
    for t in triangles:
        triangles_dict[t[0]].add(t[1])
        triangles_dict[t[0]].add(t[2])
    for id, _ in enumerate(point_positions):
        for c_id in triangles_dict[id]:
            if not graph.has_edge(c_id, id):
                 graph.add_edge(c_id, id) # add the edge in the graph 
    return graph


def ini_graph_attributes(graph, point_positions, max_corners, min_corners, instance_ids, bb_volume):
    'set attributes to each voxel by checking the associated bounding box'
    'min max corner of each instance, leb(corner)==instance_ids; one box one instance'
    bb_occupancy = is_within_bb(point_positions, min_corners[:, None], max_corners[:, None])
    activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(point_positions))]
    num_BBs_per_point = bb_occupancy.sum(axis=0)
    inst_per_point = np.ones(len(point_positions), dtype=np.int) * -2
    for id, _ in enumerate(point_positions):
        if num_BBs_per_point[id] == 1:  # point in only one box
            bb_id = activations_per_point[id][0,0] 
            inst_per_point[id] = instance_ids[bb_id]
            graph.set_vertex_attribute(id, 'category', 1)  # 1 means univocal set
        elif num_BBs_per_point[id] == 0: # point in no box, more likely background
            inst_per_point[id] = -1  
            graph.set_vertex_attribute(id, 'category', 0)  # 0 means background
        else:  # more than 1 box accociated
            bb_ids = activations_per_point[id].reshape(-1) 
            # ////////
            # smallest_box_id = np.argmin(bb_volume[bb_ids])
            # inst_id = instance_ids[bb_ids[smallest_box_id]]
            # inst_per_point[id] = inst_id # smallest bb
            # graph.set_vertex_attribute(id, 'category', 1)
            # ///////
            remove_box_id = [] # if box i in included in box j, remove box b 
            for i, _ in enumerate(bb_ids): 
                for j, _ in enumerate(bb_ids):
                    if i != j: 
                        intersect_volumn = box_intersect_volume(max_corners[i], min_corners[i], max_corners[j], min_corners[j])
                        if intersect_volumn/bb_volume[i]>0.8 and intersect_volumn/bb_volume[j]<0.6: # box i is basicly in box j 
                            remove_box_id.append(j)
            for r in remove_box_id:
                bb_ids = np.delete(bb_ids, r)
            if len(bb_ids)==1:
                inst_per_point[id] = instance_ids[bb_ids[0]]
                graph.set_vertex_attribute(id, 'category', 1)  # 1 means univocal set
            else:
                assert len(bb_ids)>1
                graph.set_vertex_attribute(id, 'category', 2) # 2 means equivocal
    return graph, inst_per_point


def propogate(graph, inst_per_point):
    e_queue = Queue()   
    # put equivocal has univocal neighbour into the queue
    for id in range(len(inst_per_point)):
        if graph.get_vertex_attribute(id, 'category') == 2:
            for nei in graph.neighbors(id):
                if graph.get_vertex_attribute(nei, 'category') == 1:
                    e_queue.put((id, inst_per_point[nei])) # a tuple, id and the instance id should be 
                    break
    # then propogate and add new equivocal in the queue
    while not e_queue.empty():
        element = e_queue.get()
        graph.set_vertex_attribute(element[0], 'category', 1) # set to univocal
        inst_per_point[element[0]] = element[1]
        for nei in graph.neighbors(element[0]):     # add all the equi neighbours to the queue
            if graph.get_vertex_attribute(nei, 'category') == 2:
                if (nei,element[1]) not in e_queue.queue:
                    e_queue.put((nei,element[1]))
    return inst_per_point


def super_point_smooth(unique_vox_segments, segments, inst_per_point):
    'Generate the super-point inst label by smooth the point inst label within the super-points '
    inst_per_seg = np.ones(len(unique_vox_segments), dtype=np.int) * -2  # instance id of each over-segment
    for i, seg_id in enumerate(unique_vox_segments):
        seg_mask = seg_id == segments # segments is scene[segments]
        ins_ids = inst_per_point[seg_mask] # the instance ids in the segments 
        inst_per_seg[i] = stats.mode(ins_ids, keepdims=True)[0][0]  # take the mode of the inst id as this segment's inst id 
    return inst_per_seg


def generate_labels(scene, labels, cfg):
    # prepare all the variables we need
    vox_coords, vox_segments, unique_vox_segments, vox2point = voxel_switch(scene)
    semantics = labels['per_instance_semantics']
    scene_fg = (semantics > 2) & (semantics != 22)
    centers = labels['per_instance_bb_centers'][scene_fg]
    bounds = labels['per_instance_bb_bounds'][scene_fg] + 0.005
    instance_ids = labels['unique_instances'][scene_fg]
    min_corners = centers - bounds
    max_corners = centers + bounds
    bb_volume = np.prod(2 * bounds, axis=1)
    # Add noise to boxes 
    if cfg.noise>0:
        noisy_boxes = cfg.noise
        rng = np.random.default_rng(seed=3000) 
        min_corners += rng.normal(loc=0, scale=noisy_boxes/2, size= min_corners.shape) # scale is std dev
        max_corners += rng.normal(loc=0, scale=noisy_boxes/2, size= max_corners.shape) # scale is std dev
    # do the pseudo label generation 
    if cfg.h == 'algorithm':
        # call the helper functions 
        graph = generate_point_graph(scene['triangles'], scene['positions'])
        graph, inst_per_point = ini_graph_attributes(graph, scene['positions'], max_corners, min_corners, instance_ids, bb_volume)
        inst_per_point = propogate(graph, inst_per_point)
        inst_per_seg = super_point_smooth(unique_vox_segments, scene['segments'], inst_per_point)
    elif cfg.h == 'mode':
        inst_per_seg = approx_association_mode(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume)
    else:
        inst_per_seg = approx_association_min(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume)
    # sementic labels 
    segments_instances = labels['seg2inst'][unique_vox_segments]
    gt_full_sem = labels['per_instance_semantics'][segments_instances] # full supervision semantics
    gt_unlabeled = gt_full_sem == 0 # scannet has missing annotations, don't supervise on those
    fg_instances_seg = inst_per_seg > -1 # exclude bg and unknown
    gt_semantics_seg = np.zeros(len(fg_instances_seg), dtype=np.int) # we use semantics of instances only where we have instances
    gt_semantics_seg[fg_instances_seg] = labels['per_instance_semantics'][inst_per_seg[fg_instances_seg]] 
    gt_semantics_seg[inst_per_seg == -1] = 2 # background
    gt_semantics_seg[gt_unlabeled] = 0   # unlabeled 
    error(gt_semantics_seg, gt_full_sem, fg_instances_seg, gt_unlabeled)
    return inst_per_seg


def approx_association_mode(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume):
    inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
    bb_occupancy = is_within_bb(scene['positions'], min_corners[:, None], max_corners[:, None])
    activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
    num_BBs_per_point = bb_occupancy.sum(axis=0)
    inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
    for i, activ in enumerate(activations_per_point):
        if num_BBs_per_point[i] == 1:  # if point is active in one row (bb)
            bb_idx = activ[0, 0]
            inst_per_point[i] = instance_ids[bb_idx]  # add the row idx (bb idx)
        elif num_BBs_per_point[i] > 1:  # multiple BBs associated, label unknown
            box_ids = activ.reshape(-1)
            smallest_box_id = np.argmin(bb_volume[box_ids])
            inst_id = instance_ids[box_ids[smallest_box_id]]
            inst_per_point[i] = inst_id # smallest bb
    inst_per_seg_maj_pooled = np.ones(len(unique_vox_segments), dtype=np.int) * -2
    for i, seg_id in enumerate(unique_vox_segments):
        seg_mask = seg_id == scene['segments']
        ins_id = stats.mode(inst_per_point[seg_mask], keepdims = True)[0][0]
        inst_per_seg_maj_pooled[i] = ins_id
    return inst_per_seg_maj_pooled


def approx_association_min(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume):
    bb_occupancy = is_within_bb(scene['positions'], min_corners[:, None], max_corners[:, None])
    activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
    num_BBs_per_point = bb_occupancy.sum(axis=0)
    inst_per_seg_pooled = np.ones(len(unique_vox_segments), dtype=np.int) * -2
    for i, seg_id in enumerate(unique_vox_segments):
        seg_mask = seg_id == scene['segments']
        num_BBs_on_seg = num_BBs_per_point[seg_mask]
        min_BBs_on_seg = num_BBs_on_seg.min()   
        if min_BBs_on_seg == 1:
            seg_idx = np.where(num_BBs_on_seg == 1)[0][0]  # get index of point within segment
            seg_idx2point_idx = np.where(seg_mask)[0]
            point_idx = seg_idx2point_idx[seg_idx]  # get index of point within scene
            bb_idx = activations_per_point[point_idx][0, 0]  # get BB of that point
            inst_per_seg_pooled[i] = instance_ids[bb_idx]  # set the same BB for all seg points
        if min_BBs_on_seg == 0: # no BB, this is background. We set the -1
            inst_per_seg_pooled[i] = -1
    # smallest_bb_heuristic:
    undecided_segs = np.where(inst_per_seg_pooled == -2)[0]
    for undecided_seg in undecided_segs:
        seg_id = unique_vox_segments[undecided_seg]
        seg_mask = seg_id == scene['segments']
        num_BBs_on_seg = num_BBs_per_point[seg_mask]
        seg_point = num_BBs_on_seg.argmin()
        point_idx = np.where(seg_mask)[0][seg_point]
        box_ids = activations_per_point[point_idx].reshape(-1)
        smallest_box_id = np.argmin(bb_volume[box_ids])
        inst_id = instance_ids[box_ids[smallest_box_id]]  # 这里
        inst_per_seg_pooled[undecided_seg] = inst_id
    return inst_per_seg_pooled








