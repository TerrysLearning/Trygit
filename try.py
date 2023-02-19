import numpy as np

# segs = np.array([1,1,1,2,2,3,3,4,5,1])
# uni_segs, s2p = np.unique(segs, return_inverse=True)
# # find the ith segs is in which uniseg\
# i = -1
# print(uni_segs[s2p[i]])


# a = np.array([[1,2,3,4,4],[1,2,3,4,5]])
# a[:,0] = np.array([2,2])

# d = dict()
# d['a'] = [1,2,3]
# d['b'] = [3,4,5]
# d['c'] = [5,6]
# d['f'] = d.pop('c')
# print(d)

# a = set()
# a.add(1)
# a.add(2)
# a.add(3)
# c = np.array(list(a))
# b = np.array([2,3,4])
# print(b[np.array([1,2])])
# from collections import Counter

# def count_elements(array):
#     counter = Counter(array)
#     return [(key, value/len(array)) for key, value in counter.items()]

# A = np.array([1,2,3,4,5,6,6,6])
# result = count_elements(A)
# print(result)

# a = np.array([6,1,2,3,4,5,6,6,6])
# b, a2b, b2a = np.unique(a, return_index=True, return_inverse=True)
# print(a, a2b, b2a)
# print(b[b2a])
# print(len(b2a))
# print(a[a2b])
# # print(b)
# a = np.array([3000,2000,3000,3000,4000])
# c, a2c = np.unique(a, return_inverse=True)
# print(a2c)
# from torch_scatter import scatter_mean
# import torch
# src = torch.Tensor([2, 0, 1, 4, 3])
# index = torch.tensor([0, 1, 2, 2, 1])
# out = scatter_mean(src, index)
# print(out)
# import torch
# import numpy as np
# a = np.array([[0,1,2,3],[0,4,5,0],[0,4,0,7]], dtype=int)
# coords = torch.from_numpy(a)
# print((coords.max(0)[0][1:] + 1).numpy())
# spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
# print(spatial_shape)

# import torch
# dic = torch.load('scene0000_00_inst_nostuff.pth')
# # # e, c = np.unique(dic[2], return_counts=True)
# print(dic[-2][0:100])
# # print(len(dic[-2]))

# a = b = 3
# a += 1
# b -= 1

# print(a,b)

# import numpy as np

# arr = np.array([5,5,5,3,3,3,2000,1000,1000])
# unique_vals, indices = np.unique(arr, return_inverse=True)

# counts = np.searchsorted(unique_vals, arr)
# print(counts)

# a = np.array([1,1,2])
# b = np.array([2,3,4])
# dic = {'a': a, 'b':b}
# torch.save(dic, 'try.pt')

# dic = torch.load('try.pt')
# print(type(dic['a']))

# num_query = 3
# d_model = 5
# a = torch.nn.Embedding(num_query, d_model)
# B = 2
# print(a.weight.unsqueeze(0).repeat(B,1,1))
# assert torch.all(pred_bounds>=0)   
# print('passsssss')


# import copy
# def to_unique( segments): # enumeration_ids, when we have id arrays, like [0,1,2,..,n]
#     unique_segments =  copy.deepcopy(segments)
#     # make sure all segments across scenes have unique ids
#     for i in range(1, len(unique_segments)):
#         unique_segments[i] += np.max(unique_segments[i - 1]) + 1
#     unique_segments = np.concatenate(unique_segments, 0)
#     _, pooling_ids = np.unique(unique_segments, return_inverse=True)
#     return pooling_ids



# def to_unique_connects(multi_connects):
#     multi_connects = copy.deepcopy(multi_connects)
#     for i in range(1, len(multi_connects)):
#         add_value = np.max(multi_connects[i-1].keys())+1
#         for key in multi_connects[i].keys(): 
#             for j, _ in enumerate(multi_connects[i][key]):
#                 multi_connects[i][key][j][0] += add_value 
#             multi_connects[i][key+add_value] = multi_connects[i][key].pop()
#     connnects_con = {}
#     for dic in multi_connects:
#         connnects_con = connnects_con | dic
#     return connnects_con

# # ret = {}
# a = [np.array([1,2,3], np.array([4,5,6]))]
# print(np.concatenate(a,0))



# segments = np.array([[11,3,11,3],[22,3,3,4]])
# too = to_unique(segments)
# unique_segs = np.array([3,11,3,4,22])
# print(unique_segs[too])
# print(np.concatenate(segments,0))


# 'Test graph tool /////////'
# import graph_tools as graph

# g = graph.Graph(directed = False)
# # g.add_vertex(4)
# g.add_edge(1,2)
# g.add_edge(2,3)
# print(g)
# print(g.neighbors(3))
# g.set_edge_weight(1,2,99)
# print(g.get_edge_weight(1,2))
# g.set_vertex_attribute(3, 'status', 'uni')
# print(g.get_vertex_attribute(3,'status'))
# g.set_vertex_attribute(3, 'pps', 0)
# print(g.get_vertex_attribute(2,'status'), 'babababababababab')


# from generate_label import generate_point_graph, ini_graph_attributes

# print('test generate_point_graph //////////////////')
# # # 'Test geenrate graph'
# # point_positions = np.array([[0.1,0.2,0.4], [3.3, 3.7, 3.6], [4.5, 5.3, 5.4], [4.7,8.0,9.2], [0.1,0.2,0.406], [4.51,5.61, 7.79]])
# # triangles = np.array([[1,2,0],[1,2,3],[0,2,4], [2,3,5]])
# # g = generate_point_graph(triangles, point_positions)
# # print(g)

# # print('test inin_graph ////////////////')
# # centers = np.array([[4,4,4],[4,4,4]])
# # bounds = np.array([[1,1,1],[2,2,3]])
# # instance_ids = [1,2]
# # g, inst_per_point = ini_graph_attributes(g, point_positions, centers, bounds, instance_ids)
# # print(g, inst_per_point)


# print('test python /////////////////')
# a = np.array([[0],[2],[3]])
# print(a.reshape(-1))
# def get_pairs(input_array):
#     pairs = []
#     for i in range(input_array.shape[0]):
#         for j in range(i+1, input_array.shape[0]):
#             pairs.append((input_array[i], input_array[j]))
#     return pairs

# # Example usage
# my_array = np.array([1, 2, 3])
# print(get_pairs(my_array))

# vs = np.array([4,5,3])
# ids = np.array([0,1])
# vs = np.delete(vs, 2)
# print(vs[ids])

# print('//////////////////////////')
# a = np.array([9,2,3])
# b = np.array([4,1,1])
# print((a>b).all())
# print(np.absolute(a-b))

# print('////////////////////////')
# a = np.array([0,-1,2,3,-1,4,7,9,-1])
# count = np.sum(a == -1)
# print(count, 'biu')



# from queue import Queue
# # Initializing a queue
# q = Queue()
# # Adding of element to queue
# q.put(('a',2))
# q.put('c')
# q.put('c')
# q.put('c')
# # Removing element from queue
# if ('a',2) not in q.queue:
#     print('hahahbb')
# print(q.get())
# print(q.get())
# q.put('d')
# print(q.get())

# print('test scipy mode')
# import scipy.stats as stats
# a = np.array([1,3,3,4])
# m = stats.mode(a, keepdims = True)
# print(m[0][0])




# # rng = np.random.default_rng(seed=200) # always use same noise on scene
# # # by 2-sigma rule ~95% samples will be in (-noisy_boxes, noisy_boxes), with std_dev = noisy_boxes/2
# # min_corner = np.array([ 0.8388308,   0.7254205,   0.6287502])
# # max_corner = np.array([-1.1820583,  -1.9646726,  0.36740661])
# # noisy_boxes = 0.05
# # min_corner2 = min_corner + rng.normal(loc=0, scale=noisy_boxes/2, size= min_corner.shape) # scale is std dev
# # max_corner2 = max_corner + rng.normal(loc=0, scale=noisy_boxes/2, size= max_corner.shape) # scale is std dev


# # #[0.79537414 0.69200443 0.59472253] [-1.19084873 -2.02248714  0.36268418]
# # #[0.84668798 0.71472262 0.61610965] [-1.17507587 -1.93932417  0.37689969]

# # print(np.random.rand())


# print('test p3d and read ply difference')
# import os
# # from scannet import read_scene
# from plyfile import PlyData, PlyElement
# import numpy as np

# scene_name = 'scene0072_02'
# data_path = 'scans'
# # path_txt = os.path.join(data_path, scene_name, f'{scene_name}.txt')
# path_ply = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.ply')
# # positions, normals, colors, triangles = read_scene(path_ply, path_txt, align=False)

# # print(triangles[0:10])
# plydata = PlyData.read(path_ply)
# wrong_labels = np.load('wronge_label_ids.npz')
# print(wrong_labels.files)

# # print(plydata['face'][0:10])

# bb_ids = [1,2,3,4]
# for i, _ in enumerate(bb_ids): 
#     for j, _ in enumerate(bb_ids):
#         if i != j: 
#             # intersect_volumn = box_intersect_volume(max_corners[i], min_corners[i], max_corners[j], min_corners[j])
#             # if intersect_volumn/bb_volume[i]>0.8: #and intersect_volumn/bb_volume[j]<0.6: # box i is basicly in box j 
#             #     remove_box_id.append(j)
#             if bb_volume[i] / bb_volume[j] < 1:
#                 remove_box_id.append(j)
# bb_ids_new= [] 
# for x in bb_ids:
#     if x not in remove_box_id:
#         bb_ids_new.append(x)

# import torch
# import numpy as np
# input = torch.from_numpy(np.array([[1,2,3],[4,5,6]]))
# target = torch.from_numpy(np.array([[9,4,3],[4,5,8]]))
# a = torch.sum(torch.mul(input, target),axis=1)
# b = torch.norm(input.float(), p=2, dim=1)
# c = torch.norm(target.float(), p=2, dim=1)
# print(a)
# print(b)
# print(c)
# print(a/torch.mul(b,c))


# loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
# # #loss = torch.nn.CrossEntropyLoss()
# ## input = torch.randn(3, requires_grad=True)
# # # target = torch.empty(3).random_(2)


# output = loss(input, target)
# print(output)
# # # output.backward()
# # print(output)


# # Example of target with class indices
# # loss = torch.nn.CrossEntropyLoss()
# # input = torch.randn(3, 5, requires_grad=True)
# # target = torch.empty(3, dtype=torch.long).random_(5)
# # output = loss(input, target)
# # output.backward()
# # Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# print(input, target)
# output = loss(input, target)
# print(output)
# output.backward()


# target = torch.from_numpy(np.array([1.0, 0.2, 3.4, 5.6, -0.7]))
# print(torch.sort(target))
# print(torch.argsort(target))


# a = np.array([99,88,76])
# b = np.array([1,0])
# print(a[b])


# x = np.array([99,97,98])
# # print(np.bincount(x), '////////////////')


# print(np.concatenate([a, b, x]))



# def NMS_clustering(boxes, cluster_th=0.5):
#     # boxes should be a list of 3D boxes [box_score, min_corner,max_corner], higher scores for better boxes
#     remaining_boxes_indices = torch.argsort(-boxes[:, 0]) # from biggest to smallests
#     # remove score component
#     boxes = boxes[:, 1:]
#     cluster_representant = []
#     clusters = []
#     cluster_heatmaps = []
#     while len(remaining_boxes_indices) > 0:
#         remaining_boxes = boxes[remaining_boxes_indices]
#         cluster_heatmap = torch_IOUs(remaining_boxes[0], boxes)
#         cluster_heatmap[remaining_boxes_indices[0]] = 1
#         cluster_heatmaps.append(cluster_heatmap)
#         ious = cluster_heatmap[remaining_boxes_indices]
#         iou_mask = ious <= cluster_th
#         cluster_representant.append(remaining_boxes_indices[0])
#         clusters.append(remaining_boxes_indices[~iou_mask])
#         remaining_boxes_indices = remaining_boxes_indices[iou_mask]

#     return torch.Tensor(cluster_representant).long(), clusters, torch.stack(cluster_heatmaps,0)

# pred_offsets = torch.randn(5, 3, requires_grad=True)
# gt_offsets = torch.randn(5, 3, requires_grad=False)
# loss = torch.sum(torch.abs(pred_offsets-gt_offsets), axis=1)
# print(pred_offsets)
# loss.backward()
# print(pred_offsets)


# pred_array = pred_offsets.detach().numpy()
# gt_array = gt_offsets.detach().numpy()
# error_rank = np.argsort(-error_array)
# print(error_rank, error_array)
# wrong_ids = np.array([1,2])
# intersect = np.intersect1d(error_rank[0:len(wrong_ids)], wrong_ids, assume_unique=True)
# print(len(intersect)/len(wrong_ids))


# # pred_semantics = torch.randn(5, requires_grad=True)
# # gt_semantics = torch.randn(5, requires_grad=True)
# # pred_array = pred_semantics.detach().numpy()
# # gt_array = gt_semantics.detach().numpy()
# # error_array = np.absolute(pred_semantics-gt_semantics)
# # error_rank = np.argsort(error_array)
# # semantics_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
# # semantics_loss = semantics_loss(pred_semantics, gt_semantics)


# a = np.array([1,1,3])
# print(a.shape)


# loss = torch.sum(torch.abs(pred_offsets - gt_offsets), axis=1)

# a = np.array([[2,2,3],[6,3,4],[3,4,5]])
# b = np.array([[2,3,4]])
# print(a[:,None].shape)
# print(8640/8)

# p1 = np.array([0.53242141, 4.51727343, 0.26304942])
# p2 = np.array([-3.41029577, 1.22604848, 0.19869942])

# p3 = np.array([0.53613579, 4.47075367, 0.7883808])

# Rt = np.array([[ 0.945519, 0.325568, 0, -5.38439],
#                [-0.325568, 0.945519, 0, -2.87178],
#                [ 0,  0,  1,  -0.06435 ],
#                [ 0,  0,  0,  1]])


# def hom_transfer(p, Rt):
#     p_ = np.concatenate((p, np.array([1])))
#     return Rt@p_

# print(hom_transfer(p3,Rt))

# ps = np.vstack([p1, p3])


# import det_box_load

# max1 = np.array([1,1,1])
# max2 = np.array([1,1,1])
# min1 = np.array([0,0,0])
# min2 = np.array([0,0,0])660,

# from itertools import product

# def possible_arrays(x1, x2, y1, y2, z1, z2):
#     first_elements = [x1, x2]
#     second_elements = [y1, y2]
#     third_elements = [z1, z2]
#     return [list(tup) for tup in product(first_elements, second_elements, third_elements)]

# x1, x2, y1, y2, z1, z2 = 1, 2, 3, 4, 5, 6
# print(possible_arrays(x1, x2, y1, y2, z1, z2))

# a = np.array([1,2])
# b = np.array([0,1,0,1,1])
# print(a[b])

# a = np.array([99,88,42,45,88])
# b, b2a = np.unique(a, return_inverse=True)
# print(b)
# ins = np.array([1,2,4])
# print(b2a[ins])
# print(b[b2a[ins]])

# sem_acc = np.load('useless/sem_acc.npy')
# wrong_ids = np.load('useless/wrong_ids.npy')

# print(len(sem_acc))
# print(len(wrong_ids))


# # print(sem_acc[0:10])
# print(wrong_ids[0:10])
# print(len(sem_acc[wrong_ids][0:10]))

# a = np.array([1,2,3,4,5,6,6,7,2,3,1,0,9])
# b = np.isin(a, np.array([2,3,4]))
# print(np.mean(a))
# print(np.mean(a[b]) , np.mean(a[~b]))


# a = np.array([[1,2,3,1,1,1],[4,5,6,4,3,8],[1,3,5,1,7,2]])
# print(a[:,3])



    





