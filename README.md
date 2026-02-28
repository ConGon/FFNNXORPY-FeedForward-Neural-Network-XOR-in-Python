XOR Neural Network with Visualization

Author: ConGon
Date: 2026-02-27

This project implements a simple feedforward neural network (FFNN) to learn the XOR function and provides visualizations of training loss, final predictions, and prediction evolution over epochs.

Table of Contents

Features

Requirements

File Structure

Usage

Customization

License

Features

Train a small neural network on the XOR dataset.

Visualize training loss over epochs.

Compare final predicted outputs against actual outputs.

Track prediction evolution across epochs.

Automatically saves graphs to an output_graphs folder.

Requirements

Python 3.8+

Libraries:

numpy

matplotlib

Install dependencies with:

pip install numpy matplotlib
File Structure
.
├── ffnnxorp.py          # Main XOR neural network script
├── utils
│   └── visualization.py # Visualization utilities
└── output_graphs        # Generated graphs will be saved here
Usage

Clone the repository:

git clone <repository_url>
cd <repository_folder>

Run the neural network script:

python ffnnxorp.py

After training, the program will print the final predicted outputs:

Final Predicted Output:
[[0.001]
 [0.998]
 [0.997]
 [0.002]]

Graphs will be automatically saved in the output_graphs folder:

training_loss.png
 — Loss over epochs.

final_predictions.png
 — Comparison of actual vs predicted outputs.

epoch_predictions.png
 — How predictions evolved during training.

Customization

Neurons and learning rate: Modify input_neurons, hidden_neurons, output_neurons, and learning_rate in ffnnxorp.py.

Epochs: Change the epochs variable to train longer or shorter.

Output folder: Modify the output_folder variable to change where graphs are saved.

License

This project is open-source and free to use.
