import os
# os.environ["OMP_NUM_THREADS"] = "16" 
import configargparse
import numpy as np

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default='scene0000_00',
                        help='data directory')
    parser.add_argument("--h", type=str, default='min', help='superpoint heuristic')
    parser.add_argument("--n", type=float, default=0.0, help='noise value')
    parser.add_argument("--v", type=bool, default=False, help='visualise wrong labels')
    parser.add_argument("--d", type=bool, default=False, help='add dectetion bounding box')
    parser.add_argument("--a", type=bool, default=False, help='add data argumentation')
    return parser


def get_config(args=None):
    parser = config_parser()
    cfg = parser.parse_args(args)
    return cfg


if __name__ == '__main__':
    print(get_config())


