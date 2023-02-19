import cfg_loader
import scannet 
import numpy as np
import get_item
import generate_label
import time

'''
keys of ret: 
    ['vox_coords', 'vox_segments', 'vox_features', 'scene', 'vox_world_coords', 
    'vox2point', 'point2vox', 'input_location', 'seg2point', 'seg2vox', 'pred2point', 
    'labels', 'pseudo_inst', 'fg_instances', 'gt_bb_bounds', 'gt_bb_offsets', 'gt_semantics']

keys of scene:
    dict_keys(['name', 'positions', 'normals', 'colors', 'segments'])

keys of labels:
    dict_keys(['semantics', 'instances', 'centers', 'center_distances', 'bb_centers', 
    'bb_offsets', 'bb_bounds', 'seg2inst', 'bb_center_distances', 'bb_radius', 'unique_instances', 
    'per_instance_semantics', 'per_instance_bb_centers', 'per_instance_bb_bounds', 'per_instance_bb_radius'])
'''


if __name__ == '__main__':
    cfg = cfg_loader.get_config()    
    scene, labels = scannet.process_scene(cfg.scene_name, cfg.a)
    ret = generate_label.generate_labels(scene, labels, cfg)

    # t3 = time.time()


