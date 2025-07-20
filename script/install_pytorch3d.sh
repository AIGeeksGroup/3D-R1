#!/bin/bash

# Installation script for PyTorch3D
# This script provides multiple installation methods for PyTorch3D

echo "Installing PyTorch3D for 3D-R1..."

# Method 1: Install via conda (recommended)
echo "Method 1: Installing via conda (recommended)"
echo "Run the following command:"
echo "conda install pytorch3d -c pytorch3d"

# Method 2: Install via pip
echo ""
echo "Method 2: Installing via pip"
echo "Run the following command:"
echo "pip install pytorch3d"

# Method 3: Install from source (if above methods fail)
echo ""
echo "Method 3: Installing from source (if above methods fail)"
echo "Run the following commands:"
echo "git clone https://github.com/facebookresearch/pytorch3d.git"
echo "cd pytorch3d"
echo "pip install -e ."

# Check if PyTorch3D is available
echo ""
echo "Checking PyTorch3D installation..."
python -c "
try:
    import pytorch3d
    print('✓ PyTorch3D is successfully installed!')
    print(f'Version: {pytorch3d.__version__}')
except ImportError:
    print('✗ PyTorch3D is not installed or not working properly.')
    print('Please try one of the installation methods above.')
"

echo ""
echo "Installation complete!"
echo "You can now use proper 3D rendering in 3D-R1 by setting --use_pytorch3d_rendering"
