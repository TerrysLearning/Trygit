'Load detection bounding box'
import numpy as np
import os

def load(scene_name):
    '''
    Input: detection bbox obj file 
    Output: centers, bounds
    '''
    file_path = 'pred_boxes/' + scene_name + '_pred.obj'
    # file_path = 'pred_boxes/one_box.obj'
    centers = []
    bounds = []
    with open(file_path) as f:
        lines = f.readlines()
        # get the centers
        i = 0
        current_box = []
        flag = False
        while i < len(lines):
            if lines[i][0] == 'v':
                ls = lines[i].split(' ')
                v_list = []
                v_list.append(float(ls[1]))
                v_list.append(float(ls[2]))
                v_list.append(float(ls[3]))
                current_box.append(v_list)
                if (i+1)%8==0:
                    current_box = np.array(current_box)
                    c1 = np.mean(current_box[:,0])
                    c2 = np.mean(current_box[:,1])
                    c3 = np.mean(current_box[:,2])
                    center = np.array([c1, c2, c3])
                    b1 = np.max(current_box[:,0])-np.min(current_box[:,0])
                    b2 = np.max(current_box[:,1])-np.min(current_box[:,1])
                    b3 = np.max(current_box[:,2])-np.min(current_box[:,2])
                    bound = 0.5*np.array([b1, b2, b3])
                    if not flag:
                        centers = center
                        bounds = bound
                        flag = True
                    else:
                        centers = np.vstack((centers, center))
                        bounds = np.vstack((bounds, bound))
                    current_box = []
                i+=1
            else:
                break
    return centers, bounds


def get_Rt(path_txt):
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
        return Rt


def hom_transfer_all(ps, Rt):
    ps_ = []
    for p in ps:
        p_ = np.concatenate((p, np.array([1])))
        p_ = Rt@p_
        ps_.append(p_[:3])
    return np.array(ps_)


def box_iou(max_corner_1, min_corner_1, max_corner_2, min_corner_2):
    bv1 = np.prod(max_corner_1 - min_corner_1)
    bv2 = np.prod(max_corner_2 - min_corner_2)
    bb_intersect = box_intersect_volume(max_corner_1, min_corner_1, max_corner_2, min_corner_2)
    union_area = bv1 + bv2 - bb_intersect + 0.000001
    return bb_intersect / union_area


def box_intersect_volume(max_corner_1, min_corner_1, max_corner_2, min_corner_2):
    'input corners output intersect box max+min corner'
    max_c = np.minimum(max_corner_1, max_corner_2)
    min_c = np.maximum(min_corner_1, min_corner_2)
    if (max_c > min_c).all():
        return np.prod(max_c-min_c)
    else:
        return 0


def convert_box(min_corners, max_corners, scene_name):
    '''
    convert the box parameters from generated boxes to detection boxes
    select the box with the best iou with generated boxes among detections
    input: generated boxes and detection boxes load scene
    '''
    n1 = len(min_corners) # num of boxes needed (generated)
    centers_det, bounds_det = load(scene_name)
    Rt = get_Rt(os.path.join('scans', scene_name, f'{scene_name}.txt'))
    centers_det = hom_transfer_all(centers_det, Rt)
    n2 = len(centers_det)
    min_corners_det = centers_det - bounds_det
    max_corners_det = centers_det + bounds_det
    min_corners_new = []
    max_corners_new = []
    for i in range(n1):
        ious = np.array([box_iou(max_corners[i], min_corners[i], max_corners_det[j], min_corners_det[j]) for j in range(n2)])
        max_iou_id = np.argmax(ious)
        min_corners_new.append(min_corners_det[max_iou_id])
        max_corners_new.append(max_corners_det[max_iou_id])
    return np.array(max_corners_new), np.array(min_corners_new)





    

