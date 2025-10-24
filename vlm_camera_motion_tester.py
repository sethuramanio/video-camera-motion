import cv2
import base64
import json
import os
from typing import Dict, List, Tuple
import time

class VLMCameraMotionTester:
    """
    Test VLMs on camera motion detection
    """
    
    def __init__(self, video_dir: str = '/home/claude/camera_motions'):
        self.video_dir = video_dir
        self.results = []
        
    def extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        """
        Extract evenly-spaced frames from video and convert to base64
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames_b64 = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode frame as JPEG then base64
                _, buffer = cv2.imencode('.jpg', frame)
                b64_string = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(b64_string)
        
        cap.release()
        return frames_b64
    
    def create_prompt(self, motion_type: str = 'general') -> str:
        """
        Create prompt for VLM
        """
        prompts = {
            'general': """You are watching a sequence of frames from a video. Analyze the camera motion and identify what type of camera movement is being used.

The possible camera motions are:
1. PAN (camera rotating left or right on a fixed point)
2. TILT (camera rotating up or down on a fixed point)
3. ZOOM (camera lens zooming in or out, changing focal length)
4. DOLLY (camera physically moving forward or backward)
5. TRACKING (camera moving parallel to the scene)
6. STATIC (no camera movement)

Respond with ONLY the motion type and direction in this format:
MOTION_TYPE: [type]
DIRECTION: [left/right/up/down/in/out/none]
CONFIDENCE: [low/medium/high]

Example response:
MOTION_TYPE: pan
DIRECTION: right
CONFIDENCE: high

Now analyze these frames:""",
            
            'detailed': """You are a cinematography expert analyzing camera movements in a video sequence.

Look at these frames carefully and identify:
1. Is the camera moving or static?
2. If moving, what type of motion? (pan, tilt, zoom, dolly, tracking)
3. What direction is the movement?
4. Are objects changing size? (suggests zoom/dolly)
5. Are objects sliding across the frame? (suggests pan/tilt/tracking)
6. Is there parallax (foreground moving faster than background)? (suggests dolly/tracking)

Provide your analysis in this format:
MOTION_TYPE: [type]
DIRECTION: [direction]
CONFIDENCE: [low/medium/high]
REASONING: [brief explanation]""",
            
            'binary': """Look at these video frames. Is the camera moving? If yes, describe the movement briefly."""
        }
        
        return prompts.get(motion_type, prompts['general'])
    
    def test_with_anthropic_claude(self, video_path: str, ground_truth: str, api_key: str) -> Dict:
        """
        Test using Claude (Anthropic API) with vision capabilities
        Note: This is a template - requires actual API key
        """
        frames = self.extract_frames(video_path, num_frames=8)
        prompt = self.create_prompt('detailed')
        
        # This is a template for how you would call Claude API
        # Requires anthropic package and API key
        result = {
            'model': 'claude-sonnet-4-5',
            'ground_truth': ground_truth,
            'prompt': prompt,
            'response': 'API call template - requires API key',
            'correct': None,
            'video': video_path
        }
        
        return result
    
    def test_with_gemini(self, video_path: str, ground_truth: str, api_key: str) -> Dict:
        """
        Test using Google Gemini with vision
        Note: This is a template - requires actual API key and google-generativeai package
        """
        frames = self.extract_frames(video_path, num_frames=10)
        prompt = self.create_prompt('general')
        
        # Template for Gemini API call
        result = {
            'model': 'gemini-pro-vision',
            'ground_truth': ground_truth,
            'prompt': prompt,
            'response': 'API call template - requires API key',
            'correct': None,
            'video': video_path
        }
        
        return result
    
    def test_with_openai(self, video_path: str, ground_truth: str, api_key: str) -> Dict:
        """
        Test using OpenAI GPT-4V
        Note: This is a template - requires actual API key and openai package
        """
        frames = self.extract_frames(video_path, num_frames=10)
        prompt = self.create_prompt('general')
        
        # Template for OpenAI API call
        result = {
            'model': 'gpt-4-vision-preview',
            'ground_truth': ground_truth,
            'prompt': prompt,
            'response': 'API call template - requires API key',
            'correct': None,
            'video': video_path
        }
        
        return result
    
    def run_local_test(self, video_path: str, ground_truth: str) -> Dict:
        """
        Run a local test that shows what the model would see
        and provides a template response
        """
        frames = self.extract_frames(video_path, num_frames=10)
        prompt = self.create_prompt('general')
        
        # Simulate what information the model receives
        result = {
            'model': 'local_simulation',
            'ground_truth': ground_truth,
            'video_path': video_path,
            'num_frames_extracted': len(frames),
            'prompt_used': prompt,
            'note': 'This is a simulation. To test real VLMs, provide API keys.',
            'frames_preview': f'{len(frames)} frames extracted from video'
        }
        
        return result
    
    def evaluate_response(self, response: str, ground_truth: str) -> Tuple[bool, str]:
        """
        Evaluate if the model's response matches ground truth
        """
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Extract motion type from ground truth (e.g., "pan_right" -> "pan")
        gt_motion = ground_truth_lower.split('_')[0]
        
        # Check if motion type is mentioned
        if gt_motion in response_lower:
            # Check direction if applicable
            if '_' in ground_truth_lower:
                gt_direction = ground_truth_lower.split('_')[1]
                if gt_direction in response_lower:
                    return True, f"Correct: {gt_motion} {gt_direction}"
                else:
                    return False, f"Motion correct ({gt_motion}) but direction wrong"
            return True, f"Correct: {gt_motion}"
        else:
            return False, f"Incorrect: expected {gt_motion}"
    
    def run_test_suite(self, model_type: str = 'local', api_key: str = None) -> List[Dict]:
        """
        Run complete test suite on all generated videos
        """
        # Ground truth labels
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
        
        print(f"\n{'='*70}")
        print(f"Running Camera Motion Detection Test Suite")
        print(f"Model: {model_type}")
        print(f"{'='*70}\n")
        
        for video_file, ground_truth in test_cases.items():
            video_path = os.path.join(self.video_dir, video_file)
            
            if not os.path.exists(video_path):
                print(f"⚠️  Video not found: {video_path}")
                continue
            
            print(f"\nTesting: {video_file} (Ground Truth: {ground_truth})")
            print("-" * 70)
            
            # Run appropriate test based on model type
            if model_type == 'local':
                result = self.run_local_test(video_path, ground_truth)
            elif model_type == 'claude':
                result = self.test_with_anthropic_claude(video_path, ground_truth, api_key)
            elif model_type == 'gemini':
                result = self.test_with_gemini(video_path, ground_truth, api_key)
            elif model_type == 'openai':
                result = self.test_with_openai(video_path, ground_truth, api_key)
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            results.append(result)
            
            # Print result summary
            print(f"✓ Test completed for {video_file}")
            
        print(f"\n{'='*70}")
        print(f"Test Suite Complete - {len(results)} tests run")
        print(f"{'='*70}\n")
        
        return results
    
    def generate_report(self, results: List[Dict], output_path: str = '/home/claude/test_report.json'):
        """
        Generate a JSON report of test results
        """
        report = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(results),
            'results': results,
            'summary': {
                'models_tested': list(set([r.get('model', 'unknown') for r in results])),
                'camera_motions_tested': list(set([r.get('ground_truth', 'unknown') for r in results]))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {output_path}")
        
        # Also create a human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CAMERA MOTION DETECTION TEST REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Test Date: {report['test_date']}\n")
            f.write(f"Total Tests: {report['total_tests']}\n\n")
            
            f.write("Camera Motions Tested:\n")
            for motion in report['summary']['camera_motions_tested']:
                f.write(f"  - {motion}\n")
            f.write("\n")
            
            f.write("Individual Test Results:\n")
            f.write("-"*70 + "\n")
            for i, result in enumerate(results, 1):
                f.write(f"\nTest #{i}:\n")
                f.write(f"  Video: {os.path.basename(result.get('video', 'unknown'))}\n")
                f.write(f"  Ground Truth: {result.get('ground_truth', 'unknown')}\n")
                f.write(f"  Model: {result.get('model', 'unknown')}\n")
                if 'response' in result:
                    f.write(f"  Response: {result['response'][:100]}...\n")
        
        print(f"📄 Summary saved to: {summary_path}")
        
        return report

def main():
    """
    Main execution function
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        CAMERA MOTION DETECTION - VLM TESTING FRAMEWORK           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

This framework tests Vision Language Models on camera motion detection.

Generated test videos with the following camera motions:
  • Pan (left/right)
  • Tilt (up/down)  
  • Zoom (in/out)
  • Dolly (in/out)
  • Static (no motion)

To test with real VLM APIs:
  1. Get API key from provider (Anthropic/Google/OpenAI)
  2. Uncomment and configure the appropriate test function
  3. Install required packages (anthropic, google-generativeai, openai)

Running local simulation for demonstration...
    """)
    
    # Initialize tester
    tester = VLMCameraMotionTester()
    
    # Run test suite (local simulation)
    results = tester.run_test_suite(model_type='local')
    
    # Generate report
    report = tester.generate_report(results)
    
    print("\n✅ Testing framework ready!")
    print("\nNext steps to test real VLMs:")
    print("  1. Get API keys from your VLM provider")
    print("  2. Install: pip install anthropic google-generativeai openai")
    print("  3. Modify the test functions with your API key")
    print("  4. Run: tester.run_test_suite(model_type='gemini', api_key='YOUR_KEY')")

if __name__ == "__main__":
    main()
