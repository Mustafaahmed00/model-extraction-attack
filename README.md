# Model Extraction Attack using STL-10 Dataset

This project implements a model extraction attack using knowledge distillation techniques on the STL-10 dataset.

## Project Structure
```
Model_Extraction_Attack/
├── kd.ipynb              # Knowledge distillation implementation
├── student_model.h5      # Student model weights
├── teacher_model.h5      # Teacher model weights
└── README.md            # This file
```

## Setup Instructions
1. Download the STL-10 dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stl10)
2. Place the `stl10.zip` file in the project directory
3. Run the Jupyter notebook `kd.ipynb`

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Keras
- Matplotlib

## Implementation Details
- Uses knowledge distillation for model extraction
- Implements both teacher and student models
- Evaluates model performance and fidelity

## Dataset
- STL-10 dataset (96x96 color images)
- 10 classes
- 5000 training images
- 8000 test images 