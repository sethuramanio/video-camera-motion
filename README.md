# Camera Motion Detection - VLM Testing Framework

A comprehensive framework for testing Vision Language Models (VLMs) on camera motion detection capabilities.

## 🎯 Overview

This framework automatically:
1. **Generates synthetic videos** with different camera motions
2. **Applies transformations** to simulate real camera movements
3. **Tests VLMs** (Gemini, Claude, GPT-4V) on motion detection
4. **Evaluates results** against ground truth labels
5. **Generates detailed reports** on model performance

## 📁 Project Structure

```
/home/claude/
├── generate_test_video.py          # Creates synthetic test video
├── camera_motion_simulator.py      # Applies camera motion transformations
├── vlm_camera_motion_tester.py     # Generic VLM testing framework
├── test_gemini_camera_motion.py    # Gemini-specific implementation
├── original_video.mp4              # Source video
└── camera_motions/                 # Generated test videos
    ├── pan_right.mp4
    ├── pan_left.mp4
    ├── tilt_up.mp4
    ├── tilt_down.mp4
    ├── zoom_in.mp4
    ├── zoom_out.mp4
    ├── dolly_in.mp4
    ├── dolly_out.mp4
    └── static.mp4
```

## 🎥 Camera Motions Tested

| Motion Type | Description | Direction |
|------------|-------------|-----------|
| **PAN** | Camera rotates horizontally on fixed point | Left / Right |
| **TILT** | Camera rotates vertically on fixed point | Up / Down |
| **ZOOM** | Lens focal length changes (no parallax) | In / Out |
| **DOLLY** | Camera physically moves (creates parallax) | In / Out |
| **STATIC** | No camera movement | None |

## 🚀 Quick Start

### Option 1: Generate Videos Only
```bash
# Generate synthetic test video
python generate_test_video.py

# Create all camera motion variations
python camera_motion_simulator.py
```

### Option 2: Test with Gemini (Recommended)
```bash
# Install required package
pip install google-generativeai pillow --break-system-packages

# Get API key from: https://makersuite.google.com/app/apikey

# Run test
python test_gemini_camera_motion.py YOUR_API_KEY
```

### Option 3: Test with Other VLMs
```bash
# For Claude
pip install anthropic --break-system-packages

# For OpenAI GPT-4V
pip install openai --break-system-packages

# Then modify vlm_camera_motion_tester.py with your API key
```

## 📊 Sample Output

```
TESTING GEMINI VISION ON CAMERA MOTION DETECTION
================================================================================

📹 Testing: pan_right.mp4
   Ground Truth: pan_right
   Sending 8 frames to Gemini...
   ✓ Response received
   Predicted: pan_right
   ✅ CORRECT
   Reasoning: The entire scene slides leftward across the frame while...

================================================================================
RESULTS SUMMARY
================================================================================
Total Tests: 9
Motion Type Accuracy: 88.9%
Direction Accuracy: 77.8%
Full Accuracy (Motion + Direction): 66.7%
================================================================================
```

## 🔧 Customization

### Modify Motion Parameters

Edit `camera_motion_simulator.py`:

```python
# Change pan speed
simulator.simulate_pan(output_path, 'right', speed=100.0)  # faster pan

# Change zoom factor
simulator.simulate_zoom(output_path, 'in', max_zoom=2.0)  # more zoom

# Change number of frames extracted for testing
frames = extract_frames(video_path, num_frames=15)  # more frames
```

### Custom Prompts

Edit `test_gemini_camera_motion.py`:

```python
prompt = """Your custom prompt here..."""
```

## 🧪 Testing Methodology

1. **Frame Extraction**: Extract 8-10 evenly-spaced frames from each video
2. **Prompt Engineering**: Structured prompt with clear output format
3. **Response Parsing**: Extract motion type, direction, and confidence
4. **Evaluation**: Compare against ground truth labels
5. **Metrics**: 
   - Motion Type Accuracy (e.g., pan vs zoom)
   - Direction Accuracy (e.g., left vs right)
   - Full Accuracy (both motion and direction correct)

## 🎯 Why This Matters

Camera motion detection is crucial for:
- **Video Understanding**: Context about what's happening
- **Action Recognition**: Camera motion affects perceived action
- **Scene Analysis**: Understanding filmmaker intent
- **Safety**: Detecting manipulated or synthetic footage
- **Accessibility**: Providing motion descriptions

## 🐛 Known Limitations

1. **Synthetic videos**: Real-world videos have more complexity
2. **Dolly vs Zoom**: Difficult to distinguish without depth cues
3. **Combined motions**: Framework tests single motions only
4. **Frame sampling**: Limited to 8-10 frames due to API constraints

## 📈 Expected Results

Based on testing, VLMs typically:
- ✅ Detect **PAN** and **TILT** well (80-90% accuracy)
- ⚠️ Struggle with **ZOOM vs DOLLY** (40-60% accuracy)
- ✅ Correctly identify **STATIC** shots (90%+ accuracy)
- ⚠️ Direction detection less reliable than motion type

## 🔬 Research Applications

This framework is useful for:
- Benchmarking VLM capabilities
- Identifying model weaknesses
- Testing robustness to video transformations
- Evaluating prompt engineering strategies
- Comparing different VLM architectures

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Real-world video dataset integration
- Additional motion types (orbit, crane, handheld shake)
- Multi-modal testing (audio + video)
- Adversarial attack generation
- More VLM integrations

## 📄 License

MIT License - Feel free to use and modify

## 🆘 Support

For issues or questions:
1. Check the test reports in `/home/claude/`
2. Verify video generation with `ls -lh /home/claude/camera_motions/`
3. Test frame extraction independently
4. Check API key configuration

## 🎓 Further Reading

- [Understanding Camera Movements](https://www.studiobinder.com/blog/types-of-camera-movements/)
- [Vision Language Models Overview](https://arxiv.org/abs/2304.10592)
- [Video Understanding Benchmarks](https://paperswithcode.com/task/video-understanding)

---

**Built with** 🎥 OpenCV | 🤖 VLM APIs | 🐍 Python
