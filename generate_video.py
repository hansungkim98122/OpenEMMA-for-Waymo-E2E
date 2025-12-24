import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
import io
import pickle
import tqdm
import json
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import matplotlib.patches as patches

def return_row(data_dict,order):
   """
   Return the list of concatenated view of cross-left, front-left, front, front-right, cross-right from preprocessed data
   """
   image_list = []
   calibration_list = []

   order = [o-1 for o in order] #0-first indexing
   for cam_ind in order: 
      image = data_dict['image_frames'][cam_ind].squeeze(0)
      K = data_dict['camera_intrinsic'][cam_ind][0]
      dist = data_dict['camera_intrinsic'][cam_ind][1]
      intrinsic = [K[0,0], K[1,2,],K[0,2], K[1,2]] + list(dist)
      calibration = {'intrinsic': intrinsic, 'extrinsic': data_dict['camera_extrinsic'][cam_ind], 'width': image.shape[2],'height':image.shape[1]}

      image_list.append(image)
      calibration_list.append(calibration)
   return image_list, calibration_list


def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = vehicle_pose
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = calibration['extrinsic']

  intrinsic = tf.constant(list(calibration['intrinsic']), dtype=tf.float32)
  metadata = tf.constant([
      calibration['width'],
      calibration['height'],
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)

  # if you specifically want a Python list:
  camera_image_metadata = vehicle_pose.reshape(-1).flatten().ravel().tolist() + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

def draw_points_on_image(image, points, size=6, color='r'):
    h, w = image.shape[:2]
    if color == 'r': 
        rgb = (255, 0, 0) 
    elif color =='b':
        rgb = (0, 0, 255)
    elif color == 'g':
        rgb = (0,255,0)
    else:
        NotImplementedError

    for u, v, ok in points:
        if not bool(ok):
            continue
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < w and 0 <= v_i < h:
            cv2.circle(image, (u_i, v_i), size, rgb, -1)

    return image

def load_dict_from_pickle(path: str) -> dict:
    with open(path,'rb') as f:
        return pickle.load(f)

def plot_topdown_view(future_waypoints_matrix, past_waypoints_matrix, s, visualize=False):
    future_waypoints_matrix = np.asarray(future_waypoints_matrix)
    past_waypoints_matrix   = np.asarray(past_waypoints_matrix)

    #  Square figure/axes 
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)  # square figure
    ax.set_aspect("equal", adjustable="box")

    #  Filled ego rectangle (centered at origin) 
    center_x, center_y = 0.0, 0.0
    box_width, box_height = 2.0, 4.0
    x_bottom_left = center_x - box_width / 2
    y_bottom_left = center_y - box_height / 2

    rect = patches.Rectangle(
        (x_bottom_left, y_bottom_left),
        box_width,
        box_height,
        linewidth=2,
        edgecolor="b",
        facecolor="b",   
    )
    ax.add_patch(rect)

    #  Waypoints with circle markers 
    ax.plot(
        future_waypoints_matrix[:, 1], future_waypoints_matrix[:, 0],
        "-o", markersize=2, linewidth=1, color="r"
    )
    ax.plot(
        past_waypoints_matrix[:, 1], past_waypoints_matrix[:, 0],
        "-o", markersize=2, linewidth=1, color="g"
    )

    ax.grid(True)

    #  Make the visible window square in *data coordinates* 
    # Use symmetric limits so the plot isn't tall/skinny.
    all_xy = np.vstack([
        past_waypoints_matrix[:, :2],
        future_waypoints_matrix[:, :2],
        np.array([[0.0, 0.0]])
    ])
    # all_xy columns are [x, y] in your matrices, but you plot (y,x),
    # so compute limits in plotted coords:
    xs = all_xy[:, 1]  # plotted x is y
    ys = all_xy[:, 0]  # plotted y is x

    max_extent = float(np.max(np.abs(np.concatenate([xs, ys]))))
    max_extent = max(max_extent, 6.0)  # minimum zoom so ego box isn't huge
    pad = 1.2
    lim = pad * max_extent
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if visualize:
        plt.show()

    #  Save to buffer (tight square) 
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)

    im = Image.open(buf).convert("RGB").resize((s, s))
    image_array = np.array(im).transpose((2, 0, 1))

    buf.close()
    plt.close(fig)
    return image_array


def main(args):
    id = args.id
    with open('waymo_val_segments.json', 'r') as f:
        segments_json = json.load(f) # Reads and parses JSON from a file
    
    ind_ctr = 0
    total_ind = len(segments_json[id])
    frames = []
    for segment_id in tqdm.tqdm(segments_json[id]):
        '''
        process frame
        '''
        #load preprocessed data
        data_dict = load_dict_from_pickle(args.dataset_dir + args.dataset + "/" + id + "-" + segment_id + '.pkl')

        vehicle_pose = data_dict['vehicle_pose']
        future_waypoints_matrix = data_dict['ego_future_xyz']
        past_waypoints_matrix = data_dict['ego_history_xyz']

        intent = data_dict['ego_intent']
        speed = data_dict['speed']

        #Import from OpenEMMA inference results
        if args.inference_input != 'None':
            filename = os.path.join(args.inference_input,str(ind_ctr),'pred_xyz.npy')
            pred_traj = np.load(filename)
        else:
            pred_traj = None

        #Draw the top row (side, front view cameras) with future waypoint overlay
        
        top_images_list, top_calibration_list = return_row(data_dict, [4,2,1,3,5])

        images_with_drawn_points = []
        top_row_width = 0
        for i in range(len(top_calibration_list)):
            #top_images_list[i] is ofshape (3,H,W)
            top_row_width += top_calibration_list[i]['width']
            waypoints_camera_space = project_vehicle_to_image(vehicle_pose, top_calibration_list[i], future_waypoints_matrix)
            if pred_traj is not None:
                waypoints_generated = project_vehicle_to_image(vehicle_pose, top_calibration_list[i], pred_traj)
                top_images_list[i] = draw_points_on_image(top_images_list[i], waypoints_generated, size=15,color='b')
            images_with_drawn_points.append(draw_points_on_image(top_images_list[i], waypoints_camera_space, size=15))
        top_row_image = np.concatenate(images_with_drawn_points, axis=2)
 
        #Draw the bottom row (rear view cameras) with past waypoint overlay
        bottom_images_list, bottom_calibration_list = return_row(data_dict,[8,7,6])

        images_with_drawn_points = []
        bottom_row_width = 0
        bottom_row_height = bottom_calibration_list[0]['height']
        for i in range(len(bottom_calibration_list)):
            bottom_row_width += bottom_calibration_list[i]['width']
            waypoints_camera_space = project_vehicle_to_image(vehicle_pose, bottom_calibration_list[i], past_waypoints_matrix)
            bottom_images_list[i] = draw_points_on_image(bottom_images_list[i], waypoints_camera_space, size=15, color='g')
            images_with_drawn_points.append(bottom_images_list[i])
        bottom_row_image = np.concatenate(images_with_drawn_points, axis=2)

        filler_width = (top_row_width - bottom_row_width) // 2 
        
        # Text information box
        right_filler = np.zeros((3,bottom_row_height,filler_width), dtype=np.uint8) #black pixels
        
        #Top-down view map box
        #Plot top-down diagram
        top_down_img = plot_topdown_view(future_waypoints_matrix,past_waypoints_matrix,s=min(filler_width,bottom_row_height))
        #Pad with black pixels
        left_filler = np.zeros((3, bottom_row_height, filler_width-bottom_row_height), dtype=np.uint8) #black pixels

        # Make HWC panel for OpenCV
        right_filler_hwc = np.zeros((bottom_row_height, filler_width, 3), dtype=np.uint8)

        texts = [
            f'UUID: {id}',
            f'Index: {ind_ctr}/{total_ind}',
            f'Seg: {segment_id}',
            f'Intent: {intent}',
            f'Speed: {float(speed):.2f} m/s',   # nicer formatting
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.4
        thickness = 2
        color = (255, 255, 255)  # white
        x = 12
        y = 40
        line_gap = 12
        ctr = 0
        for t in texts:
            # measure to increment y correctly
            if ctr > 0:
                font_scale = 2.4
            (tw, th), baseline = cv2.getTextSize(t, font, font_scale, thickness)
            if ctr == 0:
                th *= 2
            # draw
            cv2.putText(right_filler_hwc, t, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            # next line
            y += th + baseline + line_gap
            ctr+=1

        # Convert back to CHW because your concatenation expects CHW
        right_filler = right_filler_hwc.transpose(2, 0, 1)  # (3, H, W)


        bottom_row_image = np.concatenate([top_down_img,left_filler,bottom_row_image,right_filler],axis=2) #horizontal concatenation
        frame_img_array = np.concatenate([top_row_image,bottom_row_image],axis=1).transpose((1,2,0)) #vertical concatenation
        frames.append(frame_img_array)

        ind_ctr += 1

    if frames:
        processed_frames = []
        for frame in frames:
            frame_array = np.asarray(frame)
            if frame_array.ndim == 3 and frame_array.shape[0] == 3 and frame_array.shape[-1] != 3:
                frame_array = np.transpose(frame_array, (1, 2, 0))
            elif frame_array.ndim == 2:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
            processed_frames.append(frame_array.astype(np.uint8))
        os.makedirs('videos',exist_ok=True)
        if processed_frames:
            height, width = processed_frames[0].shape[:2]
            filename = f"videos/{id}.mp4" if args.inference_input =='None' else f"videos/{id}_inference.mp4"
            writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (width, height),
            )
            for frame in processed_frames: 
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True) #uuid of waymo e2e segment
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='val',required=False) #['val',''testing]
    parser.add_argument("--inference-input", type=str, default='None',required=False) #enter the folder directory to inferenced results for the particular id that you are calling

    args = parser.parse_args()
    main(args)