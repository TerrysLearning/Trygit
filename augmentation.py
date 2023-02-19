import numpy as np
import albumentations as A

SCANNET_ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

mix3d_albumentations_aug = A.load('mix3d_albumentations_aug.yaml', data_format="yaml")
color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
color_norm = A.Normalize(mean=color_mean, std=color_std)

# HUE aug
hue_aug = A.Compose([
    A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=60, val_shift_limit=50, p=1),
], p=1)


def rotate_mesh_90_degree(mesh):
  """ Randomly rotate the point clouds around z-axis (random angle in 0,90,180,270 degree)
  """
  # random_z_angle = [0, 0.5* np.pi, np.pi, 1.5 * np.pi][np.random.randint(0,4)]
  random_z_angle = [0, 0.5* np.pi, np.pi, 1.5 * np.pi][1]
  random_x_angle = 0
  random_y_angle = 0
  print(mesh.get_rotation_matrix_from_xyz((random_x_angle, random_y_angle, random_z_angle)))
  mesh.rotate(mesh.get_rotation_matrix_from_xyz((random_x_angle, random_y_angle, random_z_angle)))

def scale_mesh (mesh, min_scale=0.9, max_scale=1.1):
    """ Randomly scale the point cloud with a random scale value between min and max
    """
    scale = np.random.uniform (min_scale, max_scale)
    print(scale, 'value of scale')
    mesh.scale(scale, center=(0, 0, 0))

def apply_hue_aug(color):
    color = color * 255  # needs to be in [0,255]
    pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
    pseudo_image = hue_aug(image=pseudo_image)["image"]
    pseudo_image = mix3d_albumentations_aug(image=pseudo_image)["image"]
    color = np.squeeze(pseudo_image)

    # normalize color information
    pseudo_image = color[np.newaxis, :, :]
    color = np.squeeze(color_norm(image=pseudo_image)["image"])
    return color

