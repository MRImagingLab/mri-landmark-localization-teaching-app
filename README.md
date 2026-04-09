## Interactive MRI Landmark Localization (Teaching App)

This Streamlit application is an educational tool that demonstrates how a neural network can localize a target region in full-field-of-view MRI using a probability-based spatial representation.

The app is designed for teaching and visualization, helping learners understand how modern deep learning models transform spatial image evidence into a stable numerical location estimate without relying on heuristic thresholding or hand-crafted post-processing rules.

### Supported Inputs

MRI images in common formats: PNG, JPG, TIF, BMP

MATLAB .mat files containing:

image (HxW or HxWxT)

Example MRI views may include different anatomical planes and dynamic image series for demonstration purposes.

### What the App Demonstrates
• Generation of a spatial probability map from neural network outputs
• Landmark or center localization from probability-weighted spatial information
• Optional comparison with argmax-based localization to illustrate the instability of peak-only methods
• Automatically scaled region-of-interest visualization
• Intermediate feature map display for teaching model behavior
• Demonstration of the softmax sharpness parameter (β) as a control on spatial distributions

### Educational Focus
• Illustrates why probability-weighted localization can be more stable and robust than hard peak detection
• Demonstrates how AI models interpret spatial information in MRI
• Highlights fully differentiable, end-to-end learning strategies relevant to modern medical image analysis
• Supports intuitive learning of core concepts in image localization, neural network outputs, and spatial uncertainty

### Intended Use
• Teaching medical image processing and AI-based image analysis
• Demonstrating modern landmark localization concepts in MRI
• Helping students explore the differences between probabilistic and peak-based localization methods
• Supporting STEM education in biomedical imaging, engineering, and computational medicine

🔗 [Live App](https://pahhz8xqgyzbhuxmmtfa9e.streamlit.app/)







