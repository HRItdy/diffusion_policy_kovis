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

At first the problem many dependencies are gone. When I tried to install them again, it showed 'unable to correct the problem, some broken package held'. But there is no broken package listed when I call 'show', then according to the terminal print, I go to the folder `` and delete the wrong PPA.

Then the apt install is good, but apt install realsense2-camera still failed because of unsolvable issue, I used sudo aptitude install .... and deny the first plan, used the second to downgrade some dependencies.

After these, the launch file said no `ur_robot_driver`, so accroding to the ur_robot architecture, this node actually should be one executable node in devel. Because I ran `catkin clean` before so all these are deleted. 

I rerun the catkin_make, cannot find `robotiq_2f_85_description`. Then I changed to `catkin build` and there are some dependencies with wrong versions. I used `pip3 install` to correct them. After that, all good!

If encounter this issue:

```
Process RTDEPositionalController:
Traceback (most recent call last):
  File "/home/tiandy/miniforge3/envs/robodiff/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/tiandy/diffusion_robotics_code/imitation_learning/diffusion_policy/diffusion_policy/real_world/rtde_interpolation_controller.py", line 236, in run
    rtde_c = RTDEControlInterface(hostname=robot_ip)
RuntimeError: One of the RTDE input registers are already in use! Currently you must disable the EtherNet/IP adapter, PROFINET or any MODBUS unit configured on the robot. This might change in the future.
```

SSH to UR5:
`ssh root@<robot_ip>`
Replace `<robot_ip>` with robot ip address.

`netstat --all --program | grep 30004` find all process binding to port 30004. It should show like this:
```
0 *:30004                 *:*                     LISTEN      3223/URControl  
tcp        0      0 localhost:30004         localhost:51810         ESTABLISHED 3223/URControl  
tcp        0      0 localhost:30004         localhost:51839         ESTABLISHED 3223/URControl  
tcp        0     52 localhost:51839         localhost:30004         ESTABLISHED 3055/driverSensorUR
tcp6       0      0 localhost:51810         localhost:30004         ESTABLISHED 2600/java       
```
where 2600 is the GUI, stop it will trigger the restart of UR robot. And `URControl` is the UR controller. 

Run `kill 3055` kill `driverSensorUR`, and immidiately run `demo_real_robot.py`. Then it should work.

## TODO
1. Write the config file, train the diffusion policy.
2. Write the dataset conversion file to convert the current dataset into rlde dataset.
3. Use the minitrain colab file in open_x_embodiment repo to fine tune the RT1-X model.
4. Read the STPPO papers. Has been added to BiliBili and xiaohongshu shoucangjia.

