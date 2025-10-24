import cv2
import numpy as np
import os
from typing import Tuple, List

class CameraMotionSimulator:
    """
    Simulates different camera motions on input video
    """
    
    def __init__(self, input_video_path: str):
        self.input_video_path = input_video_path
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def simulate_pan(self, output_path: str, direction: str = 'right', speed: float = 50.0):
        """
        Simulate horizontal pan (camera rotating left/right on tripod)
        direction: 'left' or 'right'
        speed: pixels per second to pan
        """
        cap = cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create larger canvas to pan across
        canvas_width = self.width * 2
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_num = 0
        pixels_per_frame = speed / self.fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create extended canvas
            canvas = np.zeros((self.height, canvas_width, 3), dtype=np.uint8)
            canvas[:, :self.width] = frame
            canvas[:, self.width:] = frame  # Repeat to create seamless pan
            
            # Calculate pan offset
            if direction == 'right':
                offset = int(frame_num * pixels_per_frame)
            else:
                offset = int((self.total_frames - frame_num) * pixels_per_frame)
            
            offset = offset % self.width
            
            # Extract the panned view
            panned_frame = canvas[:, offset:offset + self.width]
            
            out.write(panned_frame)
            frame_num += 1
        
        cap.release()
        out.release()
        print(f"Created PAN {direction.upper()} video: {output_path}")
        
    def simulate_tilt(self, output_path: str, direction: str = 'up', speed: float = 30.0):
        """
        Simulate vertical tilt (camera rotating up/down on tripod)
        direction: 'up' or 'down'
        speed: pixels per second to tilt
        """
        cap = cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        canvas_height = self.height * 2
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_num = 0
        pixels_per_frame = speed / self.fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create extended canvas
            canvas = np.zeros((canvas_height, self.width, 3), dtype=np.uint8)
            canvas[:self.height, :] = frame
            canvas[self.height:, :] = frame
            
            # Calculate tilt offset
            if direction == 'up':
                offset = int(frame_num * pixels_per_frame)
            else:
                offset = int((self.total_frames - frame_num) * pixels_per_frame)
            
            offset = offset % self.height
            
            # Extract the tilted view
            tilted_frame = canvas[offset:offset + self.height, :]
            
            out.write(tilted_frame)
            frame_num += 1
        
        cap.release()
        out.release()
        print(f"Created TILT {direction.upper()} video: {output_path}")
    
    def simulate_zoom(self, output_path: str, zoom_type: str = 'in', max_zoom: float = 1.5):
        """
        Simulate zoom (lens focal length change)
        zoom_type: 'in' or 'out'
        max_zoom: maximum zoom factor
        """
        cap = cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate zoom factor
            progress = frame_num / self.total_frames
            if zoom_type == 'in':
                zoom_factor = 1.0 + (max_zoom - 1.0) * progress
            else:
                zoom_factor = max_zoom - (max_zoom - 1.0) * progress
            
            # Calculate crop window
            new_width = int(self.width / zoom_factor)
            new_height = int(self.height / zoom_factor)
            
            x_start = (self.width - new_width) // 2
            y_start = (self.height - new_height) // 2
            
            # Crop and resize
            cropped = frame[y_start:y_start + new_height, x_start:x_start + new_width]
            zoomed = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            out.write(zoomed)
            frame_num += 1
        
        cap.release()
        out.release()
        print(f"Created ZOOM {zoom_type.upper()} video: {output_path}")
    
    def simulate_dolly(self, output_path: str, direction: str = 'in', speed: float = 2.0):
        """
        Simulate dolly (camera physically moving forward/backward)
        Creates parallax effect - closer objects move more
        direction: 'in' or 'out'
        speed: zoom speed per second
        """
        cap = cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_num = 0
        max_scale = 1.5
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            progress = frame_num / self.total_frames
            
            if direction == 'in':
                scale_factor = 1.0 + (max_scale - 1.0) * progress
            else:
                scale_factor = max_scale - (max_scale - 1.0) * progress
            
            # Scale frame (simpler than real dolly, but shows the effect)
            new_width = int(self.width * scale_factor)
            new_height = int(self.height * scale_factor)
            
            scaled = cv2.resize(frame, (new_width, new_height))
            
            # Center crop
            x_start = (new_width - self.width) // 2
            y_start = (new_height - self.height) // 2
            
            if scale_factor >= 1.0:
                dollied = scaled[y_start:y_start + self.height, x_start:x_start + self.width]
            else:
                dollied = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                dollied[y_start:y_start + new_height, x_start:x_start + new_width] = scaled
            
            out.write(dollied)
            frame_num += 1
        
        cap.release()
        out.release()
        print(f"Created DOLLY {direction.upper()} video: {output_path}")
    
    def simulate_tracking(self, output_path: str, direction: str = 'right', speed: float = 40.0):
        """
        Simulate tracking shot (camera moving parallel to subject)
        Similar to pan but camera physically moves
        direction: 'left' or 'right'
        speed: pixels per second
        """
        # For synthetic video, tracking looks similar to pan
        # In real video with depth, there would be more parallax
        self.simulate_pan(output_path, direction, speed)
        print(f"Note: Tracking shot simulated as pan (limited parallax in synthetic video)")
    
    def simulate_static(self, output_path: str):
        """
        Create a static (no motion) version as control
        """
        cap = cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Created STATIC video: {output_path}")

def generate_all_motions(input_video: str, output_dir: str = '/home/claude/camera_motions'):
    """
    Generate all camera motion variations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    simulator = CameraMotionSimulator(input_video)
    
    # Dictionary of motions with ground truth labels
    motions = {
        'pan_right': ('pan_right.mp4', lambda s: s.simulate_pan(f'{output_dir}/pan_right.mp4', 'right', 50)),
        'pan_left': ('pan_left.mp4', lambda s: s.simulate_pan(f'{output_dir}/pan_left.mp4', 'left', 50)),
        'tilt_up': ('tilt_up.mp4', lambda s: s.simulate_tilt(f'{output_dir}/tilt_up.mp4', 'up', 30)),
        'tilt_down': ('tilt_down.mp4', lambda s: s.simulate_tilt(f'{output_dir}/tilt_down.mp4', 'down', 30)),
        'zoom_in': ('zoom_in.mp4', lambda s: s.simulate_zoom(f'{output_dir}/zoom_in.mp4', 'in', 1.5)),
        'zoom_out': ('zoom_out.mp4', lambda s: s.simulate_zoom(f'{output_dir}/zoom_out.mp4', 'out', 1.5)),
        'dolly_in': ('dolly_in.mp4', lambda s: s.simulate_dolly(f'{output_dir}/dolly_in.mp4', 'in', 2.0)),
        'dolly_out': ('dolly_out.mp4', lambda s: s.simulate_dolly(f'{output_dir}/dolly_out.mp4', 'out', 2.0)),
        'static': ('static.mp4', lambda s: s.simulate_static(f'{output_dir}/static.mp4')),
    }
    
    print(f"\n{'='*60}")
    print("Generating Camera Motion Variations")
    print(f"{'='*60}\n")
    
    generated_videos = {}
    
    for motion_name, (filename, generator_func) in motions.items():
        print(f"\nGenerating: {motion_name}...")
        generator_func(simulator)
        generated_videos[motion_name] = os.path.join(output_dir, filename)
    
    print(f"\n{'='*60}")
    print(f"All videos generated in: {output_dir}")
    print(f"{'='*60}\n")
    
    return generated_videos

if __name__ == "__main__":
    input_video = '/home/claude/original_video.mp4'
    generated = generate_all_motions(input_video)
    
    print("\nGenerated videos:")
    for motion, path in generated.items():
        print(f"  - {motion}: {path}")
