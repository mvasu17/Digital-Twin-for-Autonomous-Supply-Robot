"""
=============================================================
ROS2 LAUNCH FILE — hospital.launch.py
=============================================================
WHY A LAUNCH FILE?
- Starts multiple ROS2 nodes with one command
- Sets parameters, remappings, and environment variables
- Handles the Gazebo world + robot spawning sequence

WHAT THIS LAUNCHES:
1. Gazebo simulator with hospital.world
2. robot_state_publisher (publishes TF transforms from URDF)
3. spawn_entity (spawns our robot URDF into Gazebo)
4. hospital_robot_node (our A* + ML navigation node)
=============================================================
"""

import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():

    pkg_name = 'hospital_robot'
    pkg_dir  = get_package_share_directory(pkg_name)

    world_file = os.path.join(pkg_dir, 'worlds', 'hospital.world')
    urdf_file  = os.path.join(pkg_dir, 'resource', 'robot.urdf')

    # Read URDF content
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── 1. Set Gazebo model path ──
    set_gz_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            os.path.join(pkg_dir, 'models'),
            ':',
            '/usr/share/gazebo-11/models'
        ]
    )

    # ── 2. Start Gazebo with our hospital world ──
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_file,
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_init.so'
        ],
        output='screen'
    )

    # ── 3. Robot State Publisher (publishes TF from URDF) ──
    # WHY: ROS2 nodes need to know how robot links relate to each other
    # The robot_state_publisher reads the URDF and broadcasts transforms
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    # ── 4. Spawn robot into Gazebo (after 3 seconds) ──
    # WHY delay: Gazebo needs time to load the world before we spawn
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_robot',
                arguments=[
                    '-entity', 'hospital_robot',
                    '-topic', '/robot_description',
                    '-x', '-13.0',
                    '-y', '9.0',
                    '-z', '0.1'
                ],
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )

    # ── 5. Our Hospital Robot Node (after 6 seconds) ──
    # WHY delay: Robot must be spawned before navigation node starts
    hospital_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='hospital_robot',
                executable='robot_node',
                name='hospital_robot_node',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )

    return LaunchDescription([
        set_gz_model_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        hospital_node,
    ])
