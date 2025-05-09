# Model Extraction Attack using STL-10 Dataset

This project demonstrates a model extraction attack using knowledge distillation techniques. We use a pre-trained CIFAR-10 model as our target (teacher) and attempt to extract its knowledge using the STL-10 dataset, simulating a real-world black-box attack scenario.

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
2. Place the dataset in `Model_Extraction_Attack/STL-10/` directory
3. Run the Jupyter notebook `kd.ipynb`

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Implementation Details
- Black-box attack simulation using knowledge distillation
- Teacher model: Pre-trained CIFAR-10 classifier
- Student model: Smaller neural network architecture
- Knowledge transfer using soft labels and temperature scaling
- Performance metrics: accuracy, precision, recall

## Results
- Base student model accuracy: 54.33%
- Knowledge distillation improved accuracy: 61.35%
- Overall improvement: 7.02%

## Dataset
- STL-10 dataset (96x96 color images)
- 10 classes
- 5000 training images
- 8000 test images
- Used for querying the teacher model in black-box setting 