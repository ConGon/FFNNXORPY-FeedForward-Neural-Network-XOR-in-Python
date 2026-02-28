# FFNNXORPY

**FeedForward Neural Network XOR in Python**

**Author:** ConGon  
**Date:** 2026-02-27

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [File Structure](#file-structure)  
- [Usage](#usage)  
- [Customization](#customization)  
- [License](#license)  

---

## Features

- Train a small neural network on the XOR dataset.  
- Visualize training loss over epochs.  
- Compare final predicted outputs against actual outputs.  
- Track prediction evolution across epochs.  
- Automatically saves graphs to an `output_graphs` folder.  

---

## Requirements

- **Python 3.8+**  
- **Libraries:**  
  - `numpy`  
  - `matplotlib`  

Install dependencies with:

```bash
pip install numpy matplotlib

```
---

## File Structure
```text
.
├── FFNNXORPY.py          # Main XOR neural network script
├── utils
│   └── visualization.py # Visualization utilities
└── output_graphs        # Generated graphs will be saved here
```

## Usage
Clone the repository:
```bash
git clone https://github.com/ConGon/FFNNXORPY-FeedForward-Neural-Network-XOR-in-Python/edit/main/README.md
cd <repository_folder>
```

Run the neural network script:

```bash
python FFNNXORPY.py  
```

After training, the program prints the final predicted outputs:

Final Predicted Output:
```text
[[0.03 ]
[0.972]
[0.974]
[0.025]]
```

Graphs are automatically saved in the output_graphs folder:

- training_loss.png — Loss over epochs
- final_predictions.png — Comparison of actual vs predicted outputs
- epoch_predictions.png — How predictions evolved during training

## Customization
- Neurons and learning rate: Modify input_neurons, hidden_neurons, output_neurons, and learning_rate in FFNNXORPY.py.
- Epochs: Change the epochs variable to train longer or shorter.
- Output folder: Modify the output_folder variable to change where graphs are saved.

## License
This project is open-source and free to use.
