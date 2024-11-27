# AlexNet
This project is a reimplementation of the **AlexNet architecture**, originally introduced in the research paper *"ImageNet Classification with Deep Convolutional Neural Networks"* by Alex Krizhevsky et al., using **PyTorch**. The model is trained on the **CIFAR-10 dataset**, with input images resized from their original size of **32x32** to the required dimensions for AlexNet's architecture.

## Features
- **Reimplementation in PyTorch**: A faithful recreation of the AlexNet model architecture.
- **Training on CIFAR-10**: The CIFAR-10 dataset, consisting of 60,000 32x32 color images across 10 classes, has been preprocessed to fit the input size of AlexNet.
- **Custom Training Pipeline**: Includes data preprocessing, training, and evaluation scripts.
- **Links to Resources**:
  - [Original Paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Architecture Details
AlexNet consists of the following key components:
1. **Input Layer**: Accepts images resized to 227x227 (from the CIFAR-10's native size of 32x32).
2. **Convolutional Layers**:
   - Five convolutional layers with **ReLU activation**.
   - Utilizes **overlapping max-pooling** in specific layers for dimensionality reduction and feature extraction.
3. **Fully Connected Layers**:
   - Three fully connected layers for classification, with **dropout regularization**.
   - The final layer has 10 output units corresponding to the CIFAR-10 classes.
4. **Normalization**: Implements local response normalization in the early convolutional layers.
5. **Optimizer and Loss**:
   - Training is performed using **Adam optimizer** or **SGD with momentum**.
   - Uses **CrossEntropyLoss** for classification.

## Results
The model achieves competitive accuracy on the CIFAR-10 dataset, demonstrating the effectiveness of AlexNet architecture with modern training optimizations.

## License
### [MIT](./LICENSE.txt)