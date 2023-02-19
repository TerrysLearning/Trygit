# from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
import graph_tools 
from queue import Queue
import det_box_load 
import visual 

# for visualise purpose
from plyfile import PlyData, PlyElement
import os
from seg_connection import gen_seg_connection

import time


def is_within_bb(points, bb_min, bb_max):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)


def is_foreground(sem):
    return (sem > 2) & (sem != 22)


def sem_error(gt_semantics, gt_full_sem, fg_instances, gt_unlabeled):
    num_1 = 0 # miss labeled background into fg_instance
    num_2 = 0 # miss labeled fg_instance into background
    num_3 = 0 # miss labeled fg_instance into another
    segs_n = len(gt_semantics)
    unlab_n = len(np.where(gt_unlabeled==True)[0])
    lab_n = segs_n - unlab_n
    # //////////////
    wronge_label_id_1 = []
    wronge_label_id_2 = []
    wronge_label_id_3 = []
    # //////////////
    # print('total labeled', lab_n)
    for i in range(segs_n):
        if fg_instances[i] and (not is_foreground(gt_full_sem[i])) and (not gt_unlabeled[i]):
            num_1 += 1
            wronge_label_id_1.append(i)
        elif (not fg_instances[i]) and is_foreground(gt_full_sem[i]) and (not gt_unlabeled[i]):
            num_2 += 1
            wronge_label_id_2.append(i)
        elif fg_instances[i] and is_foreground(gt_full_sem[i]) and (not gt_unlabeled[i]): 
            if gt_semantics[i] != gt_full_sem[i]:
                num_3 += 1
                wronge_label_id_3.append(i)
    # save wronge ids /////////////////////
    wronge_label_id_1 = np.array(wronge_label_id_1)
    wronge_label_id_2 = np.array(wronge_label_id_2)
    wronge_label_id_3 = np.array(wronge_label_id_3)
    #//////////////////
    print('miss labels:', num_1, num_2, num_3)
    print('correct label rate', (lab_n-num_1-num_2-num_3)/lab_n)
    return wronge_label_id_1, wronge_label_id_2, wronge_label_id_3


def inst_error(ins_per_seg, gt_full_inst, gt_unlabeled, gt_full_sem):
    num_1 = 0 # miss labeled background into fg_instance
    num_2 = 0 # miss labeled fg_instance into background
    num_3 = 0 # miss labeled fg_instance into another
    segs_n = len(ins_per_seg)
    unlab_n = len(np.where(gt_unlabeled==True)[0])
    lab_n = segs_n - unlab_n
    # //////////////
    wronge_label_id_1 = []
    wronge_label_id_2 = []
    wronge_label_id_3 = []
    # //////////////
    # print('total labeled', lab_n)
    for i in range(segs_n):
        if not gt_unlabeled[i]:
            if ins_per_seg[i]!=-1 and not is_foreground(gt_full_sem[i]):
                num_1 += 1
                wronge_label_id_1.append(i)
            elif ins_per_seg[i]==-1 and is_foreground(gt_full_sem[i]):
                num_2 += 1
                wronge_label_id_2.append(i)
            elif ins_per_seg[i]!=-1 and is_foreground(gt_full_sem[i]): 
                if ins_per_seg[i] != gt_full_inst[i]:
                    num_3 += 1
                    wronge_label_id_3.append(i)
    # save wronge ids /////////////////////
    wronge_label_id_1 = np.array(wronge_label_id_1)
    wronge_label_id_2 = np.array(wronge_label_id_2)
    wronge_label_id_3 = np.array(wronge_label_id_3)
    #//////////////////
    print('miss labels:', num_1, num_2, num_3)
    print('correct label rate', (lab_n-num_1-num_2-num_3)/lab_n)
    return wronge_label_id_1, wronge_label_id_2, wronge_label_id_3


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
    visual.visual_points(input_points)
    assert 0
    point2vox = nbrs.kneighbors(vox_coords_unique, return_distance=False)
    point2vox = point2vox.reshape(-1) # point2vox maps points to scene: num_points->num_voxels
    vox_segments = scene['segments'][point2vox]  
    unique_vox_segments, seg2vox = np.unique(vox_segments, return_inverse = True) 
    return vox_coords, vox_segments, unique_vox_segments, vox2point, seg2vox, vox2point


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
            remove_box_id = [] # if box i in included in box j, remove box b 
            for i, id_i in enumerate(bb_ids): 
                for j, id_j in enumerate(bb_ids):
                    if i != j: 
                        if bb_volume[id_i] / bb_volume[id_j] < 0.5:
                            remove_box_id.append(id_j)
            bb_ids_new= [] 
            for x in bb_ids:
                if x not in remove_box_id:
                    bb_ids_new.append(x)
            if len(bb_ids_new)==1:
                inst_per_point[id] = instance_ids[bb_ids_new[0]]
                graph.set_vertex_attribute(id, 'category', 1)  # 1 means univocal set
            else:
                assert len(bb_ids_new)>1
                graph.set_vertex_attribute(id, 'category', 2) # 2 means equivocal
    return graph, inst_per_point


def super_point_ini(graph, unique_vox_segments, segments, inst_per_point):
    'addtional super-point smooth process in the begining '
    for seg_id in unique_vox_segments:
        seg_mask = seg_id == segments # segments is scene[segments]
        ins_ids = inst_per_point[seg_mask] # the instance ids in the segments 
        if np.sum(ins_ids!=-2) > 0:
            seg_inst = stats.mode(ins_ids, keepdims=True)[0][0]
            inst_per_point[seg_mask] = seg_inst
            for index in np.where(seg_mask==True)[0]:
                graph.set_vertex_attribute(index, 'category', 1)
        else:
            inst_per_point[seg_mask] = -2 
    return graph, inst_per_point


def propogate(graph, inst_per_point):
    e_queue = Queue()   
    # put equivocal has univocal neighbour into the queue
    for id in range(len(inst_per_point)):
        if graph.get_vertex_attribute(id, 'category') == 2:
            for nei in graph.neighbors(id):
                if graph.get_vertex_attribute(nei, 'category') != 2:
                    e_queue.put((id, inst_per_point[nei])) # a tuple, id and the instance id should be 
                    break
    # then propogate and add new equivocal in the queue
    while not e_queue.empty():
        element = e_queue.get()
        graph.set_vertex_attribute(element[0], 'category', 1) # set to univocal
        inst_per_point[element[0]] = element[1]
        graph.set_vertex_attribute(element[0], 'category', 1)
        for nei in graph.neighbors(element[0]):     # add all the equi neighbours to the queue
            if graph.get_vertex_attribute(nei, 'category') == 2:
                if (nei,element[1]) not in e_queue.queue:
                    e_queue.put((nei,element[1]))
    return inst_per_point



def super_point_smooth(unique_vox_segments, segments, inst_per_point):
    'Generate the super-point inst label by smooth the point inst label within the super-points '
    inst_per_seg = np.ones(len(unique_vox_segments), dtype=np.int) * -2  # instance id of each over-segment
    n_uni_segs = []
    for i, seg_id in enumerate(unique_vox_segments):
        seg_mask = seg_id == segments # segments is scene[segments]
        ins_ids = inst_per_point[seg_mask] # the instance ids in the segments 
        inst_per_seg[i] = stats.mode(ins_ids, keepdims=True)[0][0]  # take the mode of the inst id as this segment's inst id 
        if len(np.unique(ins_ids))!=1:
            n_uni_segs.append(i)
    return inst_per_seg, n_uni_segs


def generate_labels(scene, labels, cfg):
    # prepare all the variables we need
    vox_coords, vox_segments, unique_vox_segments, vox2point, seg2vox, vox2point = voxel_switch(scene)
    # #///////
    # ply_name = 'scans/' + scene['name'] + '/' + scene['name'] + '_vh_clean_2.ply' 
    # visual.visual_some_segments(ply_name, scene['segments'], [635,563,542,637,539], unique_vox_segments)
    # #///////
    semantics = labels['per_instance_semantics']
    scene_fg = (semantics > 2) & (semantics != 22)
    centers = labels['per_instance_bb_centers'][scene_fg]
    bounds = labels['per_instance_bb_bounds'][scene_fg] + 0.001
    instance_ids = labels['unique_instances'][scene_fg]
    min_corners = centers - bounds
    max_corners = centers + bounds
    if cfg.d:
        max_corners, min_corners = det_box_load.convert_box(min_corners, max_corners, scene['name'])
        bounds = 0.5*(max_corners - min_corners)
        visual.visual_boxes_obj(max_corners, min_corners, 'det_boxes')
    else:
        visual.visual_boxes_obj(max_corners, min_corners, 'gen_boxes')
    bb_volume = np.prod(2 * bounds, axis=1)
    # Add noise to boxes 
    if cfg.n>0:
        noisy_boxes = cfg.n
        rng = np.random.default_rng(seed=3000) 
        min_corners += rng.normal(loc=0, scale=noisy_boxes/2, size= min_corners.shape) # scale is std dev
        max_corners += rng.normal(loc=0, scale=noisy_boxes/2, size= max_corners.shape) # scale is std dev
    # do the pseudo label generation 
    if cfg.h == 'algorithm':
        # call the helper functions 
        graph = generate_point_graph(scene['triangles'], scene['positions'])
        seg2point = seg2vox[vox2point]
        print(gen_seg_connection(unique_vox_segments, graph, scene['segments'], seg2point)) # ////////////////////////////
        graph, inst_per_point = ini_graph_attributes(graph, scene['positions'], max_corners, min_corners, instance_ids, bb_volume) 
        inst_per_point = propogate(graph, inst_per_point)
        inst_per_seg, n_uni_segs = super_point_smooth(unique_vox_segments, scene['segments'], inst_per_point)
    elif cfg.h == 'mode':
        inst_per_seg = approx_association_mode(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume)
    else:
        inst_per_seg = approx_association_min(scene, unique_vox_segments, max_corners, min_corners, instance_ids, bb_volume)
    # sementic labels 
    segments_instances = labels['seg2inst'][unique_vox_segments]
    gt_full_sem = labels['per_instance_semantics'][segments_instances] # full supervision semantics
    gt_full_inst = labels['unique_instances'][segments_instances]
    gt_unlabeled = gt_full_sem == 0 # scannet has missing annotations, don't supervise on those
    fg_instances_seg = inst_per_seg > -1 # exclude bg and unknown
    gt_semantics_seg = np.zeros(len(fg_instances_seg), dtype=np.int) # we use semantics of instances only where we have instances
    gt_semantics_seg[fg_instances_seg] = labels['per_instance_semantics'][inst_per_seg[fg_instances_seg]] 
    gt_semantics_seg[inst_per_seg == -1] = 2 # background
    gt_semantics_seg[gt_unlabeled] = 0   # unlabeled 
    print(gt_semantics_seg[0:10])
    # w1, w2, w3 = sem_error(gt_semantics_seg, gt_full_sem, fg_instances_seg, gt_unlabeled)
    w1, w2, w3 = inst_error(inst_per_seg, gt_full_inst, gt_unlabeled, gt_full_sem)
    all_loss = np.concatenate([w1, w2, w3])
    if cfg.v:
        visualise_wrong_label(w1, w2, w3, scene['name'], unique_vox_segments, scene['segments'], cfg)
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



def visualise_wrong_label(w1, w2, w3, scene_name, unique_vox_segements, segments, cfg):
    data_path = 'scans'
    path_ply = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.ply')
    plydata = PlyData.read(path_ply)
    for id, seg in enumerate(unique_vox_segements):
        if id in w1:
            seg_mask = segments==seg
            plydata = set_vertex_color(plydata, seg_mask, [255,0,0])  #red
        if id in w2:
            seg_mask = segments==seg
            plydata = set_vertex_color(plydata, seg_mask, [0,255,0])  #green
        if id in w3:
            seg_mask = segments==seg
            plydata = set_vertex_color(plydata, seg_mask, [0,0,255])  #green
    if cfg.h == 'algorithm':
        out_file = scene_name + '_wrong_labels.ply'
    elif cfg.h == 'mode':
        out_file = scene_name + '_wrong_labels_mode.ply'
    with open(out_file, mode='wb') as f: 
        PlyData(plydata, text=True).write(f)


def set_vertex_color(plydata, mask, color):
    for i, bool in enumerate(mask):
        if bool:
            plydata['vertex'][i][3] = color[0]
            plydata['vertex'][i][4] = color[1]
            plydata['vertex'][i][5] = color[2]
    return plydata



# def add_bbox_ply(path_ply, centers, bounds):




