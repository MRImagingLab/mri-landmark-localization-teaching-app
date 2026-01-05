## Cine MRI Centroid Predictor (Teaching App)

This Streamlit application demonstrates how a neural network localizes the center of the cardiac region in full–field-of-view (FOV) cine cardiac MRI using a probability-based spatial representation.

The app is designed for education and visualization, illustrating how modern deep learning models convert spatial evidence into a stable numerical centroid without heuristic thresholding or post-hoc rules.

### Supported Inputs

Cine CMR images: PNG / JPG / TIF / BMP

MATLAB .mat files containing:

image (HxW or HxWxT)

Cardiac views:

2CH, 3CH, 4CH, SAX (full FOV only)

### What the App Demonstrates

Spatial probability map generated from network outputs

Centroid estimation for the heart region 

Optional argmax comparison to illustrate instability of peak-based localization

Automatically scaled ROI (default: half the image size)

Intermediate feature maps (enabled for teaching)

Softmax sharpness parameter (β) explained as a distribution control

### Educational Focus

Highlights why probability-weighted localization is more stable than hard peak detection

Demonstrates robustness to different cardiac views of cine MRI

Emphasizes fully differentiable, end-to-end learning suitable for medical imaging workflows

### Intended Use

Teaching medical image processing 

Demonstrating modern centroid localization concepts in MRI and AI

Exploring differences between probabilistic and peak-based localization

🔗 [Live App](https://mri-image-quality-lab-2tvyt2uctfuzgfvj3kzptp.streamlit.app/)





