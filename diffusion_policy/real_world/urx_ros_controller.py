import os
import time
import enum
import multiprocessing as mp
import numpy as np
import urx
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class ROSURController(mp.Process):
    """
    ROS-based controller for UR robot using URX and ROS control
    """
    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            max_pos_speed=0.25,
            max_rot_speed=0.16,
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            get_max_k=128,
            ):
        super().__init__(name="ROSURController")
        # Store parameters
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
    
        # verify
        assert 0 < frequency <= 500
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        # Build ring buffer for robot state
        example = {
            'ActualTCPPose': np.zeros((6,), dtype=np.float64),
            'ActualTCPSpeed': np.zeros((6,), dtype=np.float64),
            'ActualQ': np.zeros((6,), dtype=np.float64),
            'ActualQd': np.zeros((6,), dtype=np.float64),
            'robot_receive_timestamp': 0.0
        }
        
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ROSURController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def get_robot_state(self):
        return self.ring_buffer.get()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # Initialize ROS node
        rospy.init_node('ur_controller', anonymous=True)
        
        # Initialize URX robot connection
        robot = urx.Robot(self.robot_ip)
        
        try:
            if self.verbose:
                print(f"[ROSURController] Connected to robot: {self.robot_ip}")

            # Set parameters
            if self.tcp_offset_pose is not None:
                robot.set_tcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    robot.set_payload(self.payload_mass, self.payload_cog)
                else:
                    robot.set_payload(self.payload_mass)
            
            # Main control loop
            dt = 1. / self.frequency
            curr_pose = robot.get_pose().pose_vector
            curr_t = time.monotonic()
            last_waypoint_time = curr_t

            keep_running = True
            while keep_running and not rospy.is_shutdown():
                loop_start = time.time()

                # Get current robot state
                state = {
                    'ActualTCPPose': np.array(robot.get_pose().pose_vector),
                    'ActualTCPSpeed': np.array(robot.get_tcp_speed()),
                    'ActualQ': np.array(robot.getj()),
                    'ActualQd': np.array(robot.getv()),
                    'robot_receive_timestamp': time.time()
                }
                self.ring_buffer.put(state)
                # Signal ready on first iteration
                if not self.ready_event.is_set():
                    self.ready_event.set()

                # Regulate frequency
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                if self.verbose:
                    print(f"[ROSURController] Actual frequency {1/elapsed}")

        finally:
            # Cleanup
            robot.stop()
            robot.close()
            self.ready_event.set()

            if self.verbose:
                print(f"[ROSURController] Disconnected from robot: {self.robot_ip}")