from setuptools import setup

package_name = 'rwi_agent_cloud'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='guru',
    maintainer_email='gurudeep1998@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_driver    = rwi_agent_cloud.llm_driver_node_lidar:main',
            'teleop_logger = rwi_agent_cloud.teleop_logger_node:main',
        ],
    },
)
