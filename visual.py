import pyviz3d.visualizer as viz
import numpy as np
import open3d as o3d
from open3d import geometry

import numpy as np
from plyfile import PlyData, PlyElement

def visual_boxes_obj(max_corners, min_corners, output_file):
    'Generate an obj file to visualise the boxes'
    n = len(max_corners)
    f = open(output_file+'.obj', 'w+')
    for i in range(n):
        p1 = 'v '+ str(min_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p2 = 'v '+ str(min_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        p3 = 'v '+ str(min_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p4 = 'v '+ str(min_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        p5 = 'v '+ str(max_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p6 = 'v '+ str(max_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n' 
        p7 = 'v '+ str(max_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p8 = 'v '+ str(max_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        f.write(p1)            
        f.write(p2)
        f.write(p3)
        f.write(p4)
        f.write(p5)
        f.write(p6)
        f.write(p7)
        f.write(p8)     
    for i in range(n):
        bi = i*8
        f.write('f ' + str(bi+2) + ' '+ str(bi+4) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+5) + ' '+ str(bi+2) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+1) + ' '+ str(bi+4) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+3) + ' '+ str(bi+5) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+2) + ' '+ str(bi+8) + ' '+ str(bi+4)+ '\n')
        f.write('f ' + str(bi+6) + ' '+ str(bi+2) + ' '+ str(bi+5)+ '\n')
        f.write('f ' + str(bi+6) + ' '+ str(bi+8) + ' '+ str(bi+2)+ '\n')
        f.write('f ' + str(bi+4) + ' '+ str(bi+8) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+7) + ' '+ str(bi+5) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+3) + ' '+ str(bi+8) + ' '+ str(bi+7)+ '\n')
        f.write('f ' + str(bi+7) + ' '+ str(bi+6) + ' '+ str(bi+5)+ '\n')
        f.write('f ' + str(bi+8) + ' '+ str(bi+6) + ' '+ str(bi+7)+ '\n')


def visual_points(positions):
    f = open('points.obj', 'w+')
    for i in range(len(positions)):
        p = 'v '+ str(positions[i][0])+ ' '+ str(positions[i][1])+ ' '+ str(positions[i][2])+ '\n'
        f.write(p)            
    

def visual_some_segments(input_ply, scene_segments, segment_indexs, uni_vox_segments):
    'Input segment indexes'
    plydata = PlyData.read(input_ply)
    over_seg_color = np.random.rand(len(segment_indexs),3)*255
    for index, i in enumerate(segment_indexs):
        segments_id = uni_vox_segments[i]
        p_indexs = np.where(scene_segments == segments_id)[0]
        for j in range(len(plydata['vertex'])):
            if j in p_indexs:
                if index==0:
                    plydata['vertex'][j][3] = 255
                    plydata['vertex'][j][4] = 0
                    plydata['vertex'][j][5] = 0
                else:
                    plydata['vertex'][j][3] = over_seg_color[index][0]
                    plydata['vertex'][j][4] = over_seg_color[index][1]
                    plydata['vertex'][j][5] = over_seg_color[index][2]
    with open('checkconnect.ply', mode='wb') as f: 
        PlyData(plydata, text=True).write(f)



# def visual_points(positions, triangles):
#     f = open('points.obj', 'w+')
#     for i in range(len(positions)):
#         p = 'v '+ str(positions[i][0])+ ' '+ str(positions[i][1])+ ' '+ str(positions[i][2])+ '\n'
#         f.write(p)            
#     # for i in range(len(triangles)):
#     #     f.write('f ' + str(triangles[i][0]) + ' '+ str(triangles[i][1]) + ' '+ str(triangles[i][2])+ '\n')

# def visualise():
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
# def show(self, save_path=None):
#         self.o3d_visualizer.run()
#         self.o3d_visualizer.destroy_window()
#         return
# data_bounds = 'data_info/scene0071_00_inst_bounds.npy'
# data_centers = 'data_info/scene0071_00_inst_centers.npy'
# data_bounds = 'data_info/scene0071_00_bbb.npy'
# data_centers = 'data_info/scene0071_00_bbc.npy'
# data_voxs = 'data_info/scene0071_00_vox_coords.npy'

# def viz_box():
#     bounds = np.load(data_bounds) 
#     centers = np.load(data_centers)
#     n = len(bounds)
#     v = viz.Visualizer()
#     for i in range(n):
#         v.add_bounding_box(f'Box;_{i}',
#                            position = centers[i],
#                            size=2*bounds[i],
#                            edge_width=0.01)
#     v.save('boxes')

# def viz_p_cloud():
#     voxes = np.load(data_voxs)
#     n = len(voxes)
#     n = 63600
#     v = viz.Visualizer()
#     i = 0
#     while i < n: 
#         name = 'PointClouds;'+str(i)
#         point_positions = voxes[i:i+100]
#         point_colors = (np.random.random(size=[point_positions.shape[0], 3]) * 255).astype(np.uint8)
#         v.add_points(name, point_positions, point_colors, point_size=5.0, visible=True)
#         # i += 100
#     v.save('voxes')

# def sample():
#     v = viz.Visualizer()
#     for j in range(5):
#         i = j + 1
#         name = 'PointClouds;'+str(i)
#         num_points = 3
#         point_positions = np.random.random(size=[num_points, 3])
#         point_colors = (np.random.random(size=[num_points, 3]) * 255).astype(np.uint8)
#         point_size = 25 * i
#         v.add_points(name, point_positions, point_colors, point_size=point_size, visible=False)
#     v.save('example_point_clouds')

# def visualize_scene():
    # print('Get predictions:')
    # batches, predictions = self.dataset_prediction(val_dataset, batch_size=1) 
    # print('Convert predictions to mask format:')
    # results = self.dataset_pred2result(batches, predictions)
    # vis_folder = self.results_path + f"/viz/"


