from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # params_file 作为 launch 参数暴露出来，方便你在不改 launch 的情况下切换不同数据集配置。
    default_params_file = PathJoinSubstitution([
        FindPackageShare('cuda_bevfusion_ros2'),
        'configs',
        'bevfusion_dataset.yaml',
    ])

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='BEVFusion 数据集播放节点参数文件',
    )

    bevfusion_node = Node(
        package='cuda_bevfusion_ros2',
        executable='bevfusion_dataset_node',
        name='bevfusion_dataset_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        bevfusion_node,
    ])
