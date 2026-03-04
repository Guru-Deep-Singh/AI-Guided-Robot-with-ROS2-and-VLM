from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    desc_pkg = get_package_share_directory('rwi_description')
    urdf_path = os.path.join(desc_pkg, 'urdf', 'simple_car.urdf')

    sim_pkg = get_package_share_directory('rwi_sim_gazebo')
    world_path = os.path.join(sim_pkg, 'worlds', 'road_uturn.world')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    gazebo = ExecuteProcess(
    cmd=[
        'gazebo', '--verbose', world_path,
        '-s', 'libgazebo_ros_init.so',
        '-s', 'libgazebo_ros_factory.so'
    ],
    output='screen')

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
        output='screen'
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'rwi_simple_car', '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.40'],
        output='screen'
    )

    agent_active = LaunchConfiguration('agent_active')
    use_lidar = LaunchConfiguration('use_lidar')

    agent_active_arg = DeclareLaunchArgument(
        'agent_active',
        default_value='true',
        description='Whether to start the agent node'
    )

    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='false',
        description='Whether to use the lidar-enabled agent node'
    )

    venv_python = os.path.expanduser('~/ros-with-ai/.venv/bin/python3')  # adjust

    agent_node_lidar = ExecuteProcess(
        cmd=[venv_python, '-m', 'rwi_agent_cloud.llm_driver_node_lidar'],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", agent_active, "' == 'true' and '", use_lidar, "' == 'true'"])
        )
    )

    agent_node_no_lidar = ExecuteProcess(
        cmd=[venv_python, '-m', 'rwi_agent_cloud.llm_driver_node'],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", agent_active, "' == 'true' and '", use_lidar, "' != 'true'"])
        )
    )

    return LaunchDescription([
        agent_active_arg,
        use_lidar_arg,
        gazebo,
        rsp,
        spawn,
        agent_node_lidar,
        agent_node_no_lidar
    ])