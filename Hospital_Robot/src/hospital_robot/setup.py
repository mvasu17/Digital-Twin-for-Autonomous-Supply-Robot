from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'hospital_robot'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Register package with ROS2 ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Package.xml
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        # Resource files (URDF)
        (os.path.join('share', package_name, 'resource'),
            glob('resource/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hospital Robot Team',
    maintainer_email='robot@hospital.local',
    description='Hospital stock management robot with A* navigation and ML prediction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This registers our Python node as a ROS2 executable
            # "robot_node" = command name in terminal / launch file
            # "hospital_robot.robot_node:main" = Python module:function
            'robot_node = hospital_robot.robot_node:main',
        ],
    },
)
