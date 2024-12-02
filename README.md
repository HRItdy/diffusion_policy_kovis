# Diffusion Policy

## Official diffusion policy implementation with ROS support
- ros_real_robot.py: New script to control the data collection procedure.
- real_env_ros.py: Instantiate camera and robot data collection class with ros support.
- single_realsen_ros.py: Subscribe to one individual realsense camera through ros topic.
- multi_realsense_ros.py: Subscribe to multiple realsense topics through single_realsense interface.
- urx_ros_controller.py: Remove RTDE-related content and use purely rostopic to collect data. 

## TODO
- Add usb_cam driver to achieve multi-cameras.
- 

## Record an issue (may transfer to a readme.rd in subfolder later)
I want to deploy the diffusion policy code in ubuntu20.04 with ROS1. But the ffmpeg is too stale so I install it. I guess in this step it destroyed the dependencies of ROS system.

At first the problem many dependencies are gone. When I tried to install them again, it showed 'unable to correct the problem, some broken package held'. But there is no broken package listed when I call 'show'
