#!/usr/bin/env python3
"""
MASTER SCRIPT: Complete Camera Motion Detection Pipeline

This script runs the entire pipeline:
1. Generate synthetic test video
2. Create camera motion variations
3. Test with VLM (optional)
4. Generate reports

Usage:
    python run_complete_pipeline.py                    # Generate videos only
    python run_complete_pipeline.py YOUR_GEMINI_KEY    # Generate + test with Gemini
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def run_step(step_name, command, description):
    """Run a pipeline step"""
    print(f"\n📍 STEP: {step_name}")
    print(f"   {description}")
    print("-" * 80)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {step_name}:")
        print(e.stderr)
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          CAMERA MOTION DETECTION - COMPLETE PIPELINE                 ║
║                                                                      ║
║  Automatic testing framework for VLM camera motion understanding     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Check if API key provided
    test_vlm = len(sys.argv) > 1
    api_key = sys.argv[1] if test_vlm else None
    
    if test_vlm:
        print("🔑 API Key provided - Will test with Gemini after generating videos")
    else:
        print("ℹ️  No API key provided - Will generate videos only")
        print("   To test with Gemini, run: python run_complete_pipeline.py YOUR_API_KEY")
    
    # Pipeline steps
    print_header("PIPELINE OVERVIEW")
    
    steps = [
        ("Step 1", "Generate synthetic test video"),
        ("Step 2", "Create camera motion variations (9 videos)"),
    ]
    
    if test_vlm:
        steps.append(("Step 3", "Test with Gemini Vision API"))
        steps.append(("Step 4", "Generate test reports"))
    
    for step, desc in steps:
        print(f"  {step}: {desc}")
    
    # Confirm
    print("\n" + "-"*80)
    input("Press ENTER to start pipeline... ")
    
    # Step 1: Generate original video
    print_header("STEP 1: GENERATE SYNTHETIC TEST VIDEO")
    success = run_step(
        "Generate Original Video",
        "cd /home/claude && python generate_test_video.py",
        "Creating a synthetic video with buildings and objects for testing"
    )
    
    if not success:
        print("\n❌ Pipeline failed at Step 1")
        return
    
    # Step 2: Generate motion variations
    print_header("STEP 2: CREATE CAMERA MOTION VARIATIONS")
    success = run_step(
        "Generate Motion Variations",
        "cd /home/claude && python camera_motion_simulator.py",
        "Applying transformations to simulate pan, tilt, zoom, dolly, and static shots"
    )
    
    if not success:
        print("\n❌ Pipeline failed at Step 2")
        return
    
    # List generated videos
    print("\n📹 Generated Videos:")
    try:
        import os
        video_dir = '/home/claude/camera_motions'
        videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        for i, video in enumerate(videos, 1):
            size = os.path.getsize(os.path.join(video_dir, video)) / 1024
            print(f"   {i}. {video:<20} ({size:.0f} KB)")
    except Exception as e:
        print(f"   Error listing videos: {e}")
    
    # Step 3: Test with VLM (if API key provided)
    if test_vlm:
        print_header("STEP 3: TEST WITH GEMINI VISION")
        
        # Check if package is installed
        try:
            import google.generativeai
            print("✓ google-generativeai package found")
        except ImportError:
            print("⚠️  google-generativeai not installed")
            print("\nInstalling required package...")
            install_cmd = "pip install google-generativeai pillow --break-system-packages"
            subprocess.run(install_cmd, shell=True)
        
        success = run_step(
            "Test with Gemini",
            f"cd /home/claude && python test_gemini_camera_motion.py {api_key}",
            "Sending videos to Gemini and evaluating responses"
        )
        
        if not success:
            print("\n⚠️  VLM testing failed, but videos are generated successfully")
        
        # Step 4: Show results
        if success:
            print_header("STEP 4: TEST RESULTS")
            
            try:
                import json
                with open('/home/claude/gemini_test_results.json', 'r') as f:
                    results = json.load(f)
                    summary = results.get('summary', {})
                    
                    print("📊 GEMINI PERFORMANCE SUMMARY")
                    print("-" * 80)
                    print(f"   Model: {results.get('model', 'Unknown')}")
                    print(f"   Total Tests: {results.get('total_tests', 0)}")
                    print(f"   Motion Type Accuracy: {summary.get('motion_accuracy', 'N/A')}")
                    print(f"   Direction Accuracy: {summary.get('direction_accuracy', 'N/A')}")
                    print(f"   Full Accuracy: {summary.get('full_accuracy', 'N/A')}")
                    print("-" * 80)
                    
                    # Show individual results
                    print("\n📋 Individual Test Results:")
                    for result in results.get('results', []):
                        if 'error' not in result:
                            status = "✅" if result.get('fully_correct') else "❌"
                            print(f"   {status} {result['video']}: "
                                  f"{result['predicted_motion']}_{result['predicted_direction']} "
                                  f"(GT: {result['ground_truth']})")
            except Exception as e:
                print(f"   Error reading results: {e}")
    
    # Final summary
    print_header("PIPELINE COMPLETE")
    
    print("✅ Videos Generated Successfully!")
    print(f"   Location: /home/claude/camera_motions/")
    print(f"   Count: 9 videos (8 motion types + 1 static)")
    
    if test_vlm and success:
        print("\n✅ VLM Testing Complete!")
        print(f"   Results: /home/claude/gemini_test_results.json")
    
    print("\n📚 Next Steps:")
    print("   1. Review generated videos in /home/claude/camera_motions/")
    print("   2. Check test results in gemini_test_results.json")
    print("   3. Modify prompts in test_gemini_camera_motion.py to improve accuracy")
    print("   4. Test with other VLMs (Claude, GPT-4V) using vlm_camera_motion_tester.py")
    print("   5. Use your own videos by modifying the input path")
    
    print("\n🎯 Use Cases:")
    print("   - Benchmark VLM video understanding capabilities")
    print("   - Test robustness to different camera motions")
    print("   - Identify specific weaknesses (e.g., zoom vs dolly)")
    print("   - Research prompt engineering strategies")
    
    print("\n📖 Documentation: See README.md for detailed information")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
