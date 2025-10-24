import cv2
import numpy as np

def create_test_video(output_path, duration=5, fps=30, width=1280, height=720):
    """
    Create a synthetic test video with objects and scenery
    that will be good for testing camera motion detection
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create a frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient background (sky to ground)
        for y in range(height):
            color = int(255 * (1 - y / height))
            frame[y, :] = [color, color//2, 0]  # Blue to brown gradient
        
        # Add some "buildings" at different depths
        # Far building (small)
        cv2.rectangle(frame, (100, 400), (200, 600), (100, 100, 100), -1)
        cv2.rectangle(frame, (100, 400), (200, 420), (50, 50, 50), -1)
        
        # Middle building
        cv2.rectangle(frame, (400, 300), (550, 600), (120, 120, 120), -1)
        cv2.rectangle(frame, (400, 300), (550, 330), (60, 60, 60), -1)
        
        # Near building (large)
        cv2.rectangle(frame, (800, 200), (1000, 600), (140, 140, 140), -1)
        cv2.rectangle(frame, (800, 200), (1000, 240), (70, 70, 70), -1)
        
        # Add a "tree" in foreground
        cv2.rectangle(frame, (300, 400), (320, 600), (101, 67, 33), -1)
        cv2.circle(frame, (310, 380), 50, (34, 139, 34), -1)
        
        # Add ground plane with texture
        cv2.rectangle(frame, (0, 600), (width, height), (90, 140, 90), -1)
        
        # Add some reference markers
        for i in range(0, width, 100):
            cv2.line(frame, (i, 600), (i, 620), (255, 255, 255), 2)
        
        # Add frame number for reference
        cv2.putText(frame, f'Frame: {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")

if __name__ == "__main__":
    create_test_video('/home/claude/original_video.mp4', duration=3, fps=30)
