import math
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from sklearn.neighbors import KDTree


class AppState:

    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.paused = False
        self.selected_point = None
        self.normal_vector = None
        self.verts = None  # 3D Point Cloud
        self.pc_o3d = None  # Open3D PointCloud
        self.normals_cached = None  # Cached normals
        self.kdtree = None  # KDTree for nearest neighbor search


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# Check if device has RGB camera
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()


def calculate_normals_once(points):
    """
    Calculate normal vectors using Open3D for the entire point cloud once.
    :param points: 3D point cloud as a numpy array
    :return: Open3D PointCloud object with normal vectors
    """
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(points)

    # Downsample the point cloud to reduce computational load
    pc_o3d = pc_o3d.voxel_down_sample(voxel_size=0.05 )

    # Estimate normals
    pc_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.05))

    # Orient normals to align with the camera view
    pc_o3d.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))

    return pc_o3d


def project_to_image(point, intrinsics):
    """
    Project a 3D point onto a 2D image plane using camera intrinsics.
    :param point: 3D point (x, y, z)
    :param intrinsics: Camera intrinsics from RealSense
    :return: 2D pixel coordinates (u, v)
    """
    if point[2] <= 0:  # Depth must be positive
        return None
    u = int((point[0] / point[2]) * intrinsics.fx + intrinsics.ppx)
    v = int((point[1] / point[2]) * intrinsics.fy + intrinsics.ppy)
    return u, v


def mouse_callback(event, x, y, flags, param):
    """
    Callback function for mouse events. Selects a point in the depth map.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert 2D pixel (x, y) to 1D index in the point cloud
        if state.verts is not None and state.normals_cached is not None:
            point_index = y * 640 + x
            if point_index < len(state.verts):
                state.selected_point = state.verts[point_index]
                print(f"Selected Point: {state.selected_point}")

                # Find the nearest neighbor in the downsampled point cloud
                distances, indices = state.kdtree.query(
                    state.selected_point.reshape(1, -1), k=1
                )
                nearest_index = indices[0][0]
                state.normal_vector = state.normals_cached[nearest_index]
                print(f"Nearest Point Index: {nearest_index}")
                print(f"Normal Vector: {state.normal_vector}")


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(state.WIN_NAME, mouse_callback)

try:
    while True:
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Generate point cloud
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            # Point cloud data to arrays
            v = points.get_vertices()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

            # Cache point cloud and normals
            if state.pc_o3d is None or state.normals_cached is None:
                state.pc_o3d = calculate_normals_once(verts)
                state.normals_cached = np.asarray(state.pc_o3d.normals)

                # Create KDTree for nearest neighbor search
                state.kdtree = KDTree(np.asarray(state.pc_o3d.points))

            state.verts = verts

            # Display RealSense stream
            images = np.hstack((color_image, depth_colormap))

            # Visualize the selected point and normal vector
            if state.selected_point is not None and state.normal_vector is not None:
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                # Project the selected point to 2D
                point_2d = project_to_image(state.selected_point, intrinsics)

                # Project the end of the normal vector to 2D
                normal_end = state.selected_point + state.normal_vector * 0.1  # Scale normal for visualization
                normal_end_2d = project_to_image(normal_end, intrinsics)

                if point_2d and normal_end_2d:
                    # Draw the selected point
                    cv2.circle(images, point_2d, 5, (0, 0, 255), -1)  # Red dot

                    # Draw the normal vector
                    cv2.line(images, point_2d, normal_end_2d, (255, 0, 0), 2)  # Blue line

            cv2.imshow(state.WIN_NAME, images)

        # Key handling
        key = cv2.waitKey(1)
        if key == ord("p"):
            state.paused ^= True
        elif key in (27, ord("q")):  # ESC or Q to quit
            print("Exiting...")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()