from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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

    agent_active_arg = DeclareLaunchArgument(
        'agent_active',
        default_value='true',
        description='Whether to start the agent node'
    )

    # agent_node = Node(
    #     package='rwi_agent_cloud',
    #     executable='llm_driver',
    #     output='screen',
    #     parameters=[{'use_sim_time': True}],
    #     condition=IfCondition(agent_active)
    # )

    venv_python = os.path.expanduser('~/ros-with-ai/.venv/bin/python3')  # adjust

    agent_node = ExecuteProcess(
        cmd=[venv_python, '-m', 'rwi_agent_cloud.llm_driver_node'],
        output='screen',
        condition=IfCondition(agent_active),
    )

    return LaunchDescription([
        agent_active_arg,
        gazebo,
        rsp,
        spawn,
        agent_node
    ]) 