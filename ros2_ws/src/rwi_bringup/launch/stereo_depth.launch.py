from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Define the nodes as ComposableNodes
    composable_nodes = [
        # 1. Rectify Left Image
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_left_node',
            remappings=[
                ('image', '/stereo_camera/left/image_raw'),
                ('camera_info', '/stereo_camera/left/camera_info'),
                ('image_rect', '/stereo_camera/left/image_rect')
            ],
            parameters=[{'use_sim_time': True}]
        ),
        # 2. Rectify Right Image
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_right_node',
            remappings=[
                ('image', '/stereo_camera/right/image_raw'),
                ('camera_info', '/stereo_camera/right/camera_info'),
                ('image_rect', '/stereo_camera/right/image_rect')
            ],
            parameters=[{'use_sim_time': True}]
        ),
        # 3. Disparity Node
        ComposableNode(
            package='stereo_image_proc',
            plugin='stereo_image_proc::DisparityNode',
            name='disparity_node',
            remappings=[
                ('left/image_rect', '/stereo_camera/left/image_rect'),
                ('left/camera_info', '/stereo_camera/left/camera_info'),
                ('right/image_rect', '/stereo_camera/right/image_rect'),
                ('right/camera_info', '/stereo_camera/right/camera_info'),
                ('disparity', '/stereo_camera/disparity')
            ],
            parameters=[{'approximate_sync': True, 'use_sim_time': True}]
        ),
        # 4. Point Cloud Node
        ComposableNode(
            package='stereo_image_proc',
            plugin='stereo_image_proc::PointCloudNode',
            name='point_cloud_node',
            remappings=[
                ('left/image_rect_color', '/stereo_camera/left/image_rect'),
                ('left/camera_info', '/stereo_camera/left/camera_info'),
                ('right/camera_info', '/stereo_camera/right/camera_info'),
                ('disparity', '/stereo_camera/disparity'),
                ('points2', '/stereo_camera/points2')
            ],
            parameters=[{'approximate_sync': True, 'use_sim_time': True}]
        )
    ]

    # Create the container
    container = ComposableNodeContainer(
        name='stereo_proc_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
    )

    return LaunchDescription([container])