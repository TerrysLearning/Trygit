import numpy as np
from collections import Counter

def gen_seg_connection(unique_vox_segments, graph, segments, seg2point):
    '''
    generate oversegments connections
    return dict, key=seg and values are connected segs  
    '''
    connections = dict()
    for id, seg in enumerate(unique_vox_segments):
        pids_in_seg = np.where(segments==seg)[0]
        # get neighbor points of the superpoint 
        neighbors = set()  
        for pid in pids_in_seg:
            for nei in graph.neighbors(pid):
                if nei not in pids_in_seg:
                    neighbors.add(segments[nei])
        # add connections together with the percentage
        neighbors = np.array(list(neighbors))   # ids in 'segments'
        neighbors_uni_id = seg2point[neighbors]   # ids in 'unique vox segments'
        neighbors_num = len(neighbors_uni_id)
        counter = Counter(neighbors_uni_id)
        connections[id]=[(key, value/neighbors_num) for key, value in counter.items()]
    return connections






                






