from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from multiprocessing import Event
from multiprocessing.managers import SharedMemoryManager
from threadpoolctl import threadpool_limits
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder

class Command(enum.Enum):
    SET_READY = 0
    START_RECORDING = 1
    STOP_RECORDING = 2
    RESTART_PUT = 3

class ROSRealsense:
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            namespace: str,
            shm_manager: SharedMemoryManager,
            resolution=(648,480),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node(f'ros_realsense_{namespace}', anonymous=True)
            
        self.bridge = CvBridge()
        
        # create ring buffer
        shape = resolution[::-1]
        
        examples = dict()
        examples['color'] = np.empty(
            shape=shape+(3,), dtype=np.uint8)
        # examples['depth'] = np.empty(
        #     shape=shape, dtype=np.uint16)
        # examples['infrared'] = np.empty(
        #     shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        # Create ring buffer for visualization
        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        # Create ring buffer for main data
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_READY.value,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # Initialize video writer
        self.video_writer = None
        self.recording_path = None
        self.recording_transform = recording_transform

        # Store parameters
        self.transform = transform
        self.vis_transform = vis_transform
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.verbose = verbose
        self.put_start_time = None
        self.put_idx = None

        # Create events
        self.stop_event = Event()
        self.ready_event = Event()

        # Initialize subscriber
        self.image_sub = rospy.Subscriber(
            f"{namespace}/color/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        # Set ready
        self.ready_event.set()
        
    def image_callback(self, msg):
        if self.stop_event.is_set():
            return

        # Convert ROS Image message to OpenCV image
        receive_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Create data dictionary
        data = {
            'color': cv_image,
            'camera_capture_timestamp': msg.header.stamp.to_sec(),
            'camera_receive_timestamp': receive_time,
            'timestamp': receive_time,
            'step_idx': 0  # Will be set later
        }

        # Apply transform and put to ring buffer
        put_data = data
        if self.transform is not None:
            put_data = self.transform(dict(data))

        if self.put_downsample:
            # Put frequency regulation
            if self.put_start_time is None:
                self.put_start_time = time.time()

            local_idxs, global_idxs, self.put_idx = get_accumulate_timestamp_idxs(
                timestamps=[receive_time],
                start_time=self.put_start_time,
                dt=1/self.put_fps,
                next_global_idx=self.put_idx,
                allow_negative=True
            )

            for step_idx in global_idxs:
                put_data['step_idx'] = step_idx
                put_data['timestamp'] = receive_time
                self.ring_buffer.put(put_data, wait=False)
        else:
            step_idx = int((receive_time - self.put_start_time) * self.put_fps)
            put_data['step_idx'] = step_idx
            put_data['timestamp'] = receive_time
            self.ring_buffer.put(put_data, wait=False)

        # Put to visualization buffer
        vis_data = data
        if self.vis_transform == self.transform:
            vis_data = put_data
        elif self.vis_transform is not None:
            vis_data = self.vis_transform(dict(data))
        self.vis_ring_buffer.put(vis_data, wait=False)

        # Handle recording
        if self.video_writer is not None:
            rec_data = data
            if self.recording_transform == self.transform:
                rec_data = put_data
            elif self.recording_transform is not None:
                rec_data = self.recording_transform(dict(data))
            self.video_writer.write(rec_data['color'])

        # Process commands
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['cmd'])
        except Empty:
            n_cmd = 0

        for i in range(n_cmd):
            command = dict()
            for key, value in commands.items():
                command[key] = value[i]
            
            cmd = command['cmd']
            if cmd == Command.START_RECORDING.value:
                self.start_recording_internal(
                    str(command['video_path']),
                    command['recording_start_time']
                )
            elif cmd == Command.STOP_RECORDING.value:
                self.stop_recording_internal()
            elif cmd == Command.RESTART_PUT.value:
                self.put_idx = None
                self.put_start_time = command['put_start_time']

    def start_recording_internal(self, video_path: str, start_time: float):
        if self.video_writer is not None:
            self.stop_recording_internal()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.put_fps,
            self.vis_ring_buffer.examples['color'].shape[:2][::-1]
        )
        self.recording_path = video_path

    def stop_recording_internal(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            # Convert to H264 format for better compression
            if self.recording_path is not None:
                temp_path = self.recording_path + '.temp.mp4'
                os.system(f'ffmpeg -i {self.recording_path} -c:v libx264 -crf 23 {temp_path} -y')
                os.replace(temp_path, self.recording_path)
            self.recording_path = None
            
   # Public API
    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float=-1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        if self.put_start_time is None:
            self.put_start_time = time.time()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_recording()
        self.stop_event.set()
        self.image_sub.unregister()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        pass

    @property
    def is_ready(self):
        return self.ready_event.is_set()
