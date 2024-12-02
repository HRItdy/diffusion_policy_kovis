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
