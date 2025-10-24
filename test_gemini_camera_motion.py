"""
COMPLETE EXAMPLE: Testing Gemini Vision with Camera Motion Detection

This script shows how to actually test Google's Gemini model on camera motion detection.

Prerequisites:
1. Install: pip install google-generativeai pillow --break-system-packages
2. Get API key from: https://makersuite.google.com/app/apikey
3. Set your API key in the script or as environment variable

Usage:
    python test_gemini_camera_motion.py YOUR_API_KEY
"""

import os
import sys
import cv2
import base64
from PIL import Image
import io
import json

def test_with_gemini_real(video_path: str, ground_truth: str, api_key: str) -> dict:
    """
    Actually test Gemini on camera motion detection
    """
    try:
        import google.generativeai as genai
    except ImportError:
        return {
            'error': 'google-generativeai not installed',
            'message': 'Install with: pip install google-generativeai --break-system-packages'
        }
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract 8 frames evenly spaced
    num_frames = 8
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    
    # Create prompt
    prompt = """You are watching a sequence of frames from a video in chronological order. 
    
Analyze the camera motion and identify what type of camera movement is being used.

The possible camera motions are:
- PAN: camera rotating left or right on a fixed point (horizontal rotation)
- TILT: camera rotating up or down on a fixed point (vertical rotation)
- ZOOM: camera lens zooming in or out (objects get larger/smaller, perspective stays same)
- DOLLY: camera physically moving forward or backward (creates parallax, perspective changes)
- STATIC: no camera movement

Respond ONLY in this exact format:
MOTION_TYPE: [pan/tilt/zoom/dolly/static]
DIRECTION: [left/right/up/down/in/out/none]
CONFIDENCE: [low/medium/high]
REASONING: [one sentence explaining your answer]

Example:
MOTION_TYPE: pan
DIRECTION: right
CONFIDENCE: high
REASONING: The entire scene slides leftward across the frame while object sizes remain constant, indicating a rightward pan.

Now analyze these frames:"""
    
    # Call Gemini API
    try:
        print(f"  Sending {len(frames)} frames to Gemini...")
        
        # Prepare content with frames and prompt
        content = frames + [prompt]
        
        response = model.generate_content(content)
        response_text = response.text
        
        print(f"  ✓ Response received")
        
        # Parse response
        motion_type = "unknown"
        direction = "unknown"
        confidence = "unknown"
        reasoning = ""
        
        for line in response_text.split('\n'):
            if 'MOTION_TYPE:' in line:
                motion_type = line.split(':')[1].strip().lower()
            elif 'DIRECTION:' in line:
                direction = line.split(':')[1].strip().lower()
            elif 'CONFIDENCE:' in line:
                confidence = line.split(':')[1].strip().lower()
            elif 'REASONING:' in line:
                reasoning = line.split(':')[1].strip()
        
        # Check if correct
        predicted = f"{motion_type}_{direction}" if direction != "none" else motion_type
        gt_parts = ground_truth.split('_')
        gt_motion = gt_parts[0]
        gt_direction = gt_parts[1] if len(gt_parts) > 1 else "none"
        
        motion_correct = motion_type == gt_motion
        direction_correct = direction == gt_direction
        fully_correct = motion_correct and direction_correct
        
        result = {
            'video': os.path.basename(video_path),
            'ground_truth': ground_truth,
            'predicted_motion': motion_type,
            'predicted_direction': direction,
            'confidence': confidence,
            'reasoning': reasoning,
            'motion_correct': motion_correct,
            'direction_correct': direction_correct,
            'fully_correct': fully_correct,
            'raw_response': response_text
        }
        
        return result
        
    except Exception as e:
        return {
            'video': os.path.basename(video_path),
            'ground_truth': ground_truth,
            'error': str(e),
            'success': False
        }

def run_complete_test_suite(api_key: str, video_dir: str = '/home/claude/camera_motions'):
    """
    Run complete test suite with Gemini
    """
    test_cases = {
        'pan_right.mp4': 'pan_right',
        'pan_left.mp4': 'pan_left',
        'tilt_up.mp4': 'tilt_up',
        'tilt_down.mp4': 'tilt_down',
        'zoom_in.mp4': 'zoom_in',
        'zoom_out.mp4': 'zoom_out',
        'dolly_in.mp4': 'dolly_in',
        'dolly_out.mp4': 'dolly_out',
        'static.mp4': 'static',
    }
    
    results = []
    
    print("\n" + "="*80)
    print("TESTING GEMINI VISION ON CAMERA MOTION DETECTION")
    print("="*80 + "\n")
    
    for video_file, ground_truth in test_cases.items():
        video_path = os.path.join(video_dir, video_file)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        print(f"\n📹 Testing: {video_file}")
        print(f"   Ground Truth: {ground_truth}")
        
        result = test_with_gemini_real(video_path, ground_truth, api_key)
        results.append(result)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            status = "✅ CORRECT" if result['fully_correct'] else "❌ INCORRECT"
            print(f"   Predicted: {result['predicted_motion']}_{result['predicted_direction']}")
            print(f"   {status}")
            if result.get('reasoning'):
                print(f"   Reasoning: {result['reasoning'][:100]}...")
    
    # Calculate accuracy
    successful_tests = [r for r in results if 'error' not in r]
    if successful_tests:
        motion_accuracy = sum(r['motion_correct'] for r in successful_tests) / len(successful_tests) * 100
        direction_accuracy = sum(r['direction_correct'] for r in successful_tests) / len(successful_tests) * 100
        full_accuracy = sum(r['fully_correct'] for r in successful_tests) / len(successful_tests) * 100
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(successful_tests)}")
        print(f"Motion Type Accuracy: {motion_accuracy:.1f}%")
        print(f"Direction Accuracy: {direction_accuracy:.1f}%")
        print(f"Full Accuracy (Motion + Direction): {full_accuracy:.1f}%")
        print("="*80 + "\n")
    
    # Save detailed results
    output_file = '/home/claude/gemini_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'gemini-1.5-flash',
            'total_tests': len(results),
            'results': results,
            'summary': {
                'motion_accuracy': f"{motion_accuracy:.1f}%" if successful_tests else "N/A",
                'direction_accuracy': f"{direction_accuracy:.1f}%" if successful_tests else "N/A",
                'full_accuracy': f"{full_accuracy:.1f}%" if successful_tests else "N/A"
            }
        }, f, indent=2)
    
    print(f"📊 Detailed results saved to: {output_file}\n")
    
    return results

def main():
    """
    Main function
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      GEMINI VISION - CAMERA MOTION DETECTION TEST                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    if len(sys.argv) < 2:
        print("⚠️  API KEY REQUIRED\n")
        print("Usage:")
        print("  python test_gemini_camera_motion.py YOUR_GEMINI_API_KEY")
        print("\nGet your API key from:")
        print("  https://makersuite.google.com/app/apikey")
        print("\nOr set as environment variable:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("  python test_gemini_camera_motion.py $GEMINI_API_KEY")
        
        # Check if running as demo
        print("\n" + "="*72)
        print("DEMO MODE: Running without API key")
        print("="*72)
        print("\nTo actually test Gemini:")
        print("1. pip install google-generativeai --break-system-packages")
        print("2. Get API key from https://makersuite.google.com/app/apikey")
        print("3. Run: python test_gemini_camera_motion.py YOUR_API_KEY")
        return
    
    api_key = sys.argv[1]
    
    # Check if google-generativeai is installed
    try:
        import google.generativeai as genai
        print("✓ google-generativeai package found\n")
    except ImportError:
        print("❌ google-generativeai not installed\n")
        print("Install with:")
        print("  pip install google-generativeai --break-system-packages\n")
        return
    
    # Run test suite
    results = run_complete_test_suite(api_key)
    
    print("✅ Testing complete!")
    print("\nCheck gemini_test_results.json for detailed results.")

if __name__ == "__main__":
    main()
