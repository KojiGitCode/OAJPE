# Machine Learning Datasets and Models for Grid Health Index Prediction Using Graph Attention (GAT) Models
![Single Line Diagram](images/IEEE118-Diagram.png)

## Table of Contents
1. [Key Features](#key-features)
2. [Repository Overview](#repository-overview)
3. [Getting Started](#getting-started)
    - [Pre-Trained Models](#1-pre-trained-models)
    - [Running the Code](#2-running-the-code)
4. [Results and Reproducibility](#results-and-reproducibility)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)
 
---

## Key Features
This repository includes:
- Training and testing datasets
- 25,225 power flow snapshots [^1]
- Dynamic simulation results at 40% and 100% load conditions [^2]
- Graph attention network models for frequency/angle health index
- Python code for evaluating frequency/angle health index
[^1]: Nodes are augmented to explicitly implement step-up and step-down transformers.
[^2]: Results at other load conditions (45, 50, ... 95%) are available upon request. Please email me [#Contact](#contact) for further inquiries.

## Repository Overview
This repository provides datasets, power flow snapshots, and dynamic simulation results at 40% and 100% load conditions. It includes pre-trained models using Graph Attention Networks (GAT) for frequency/angle health index prediction and Python code for evaluating these health indices. The codebase is intended for researchers and practitioners working on grid health monitoring and network analysis.

## Getting Started
### Prerequisites
- Python 3.10 or later
- PyTorch 2.0.1 or later
- PyTorch Geometric 2.3.0 or later
- 48 GB or more GPU memory is recommended when training the model.

## Results and Reproducibility
Python codes for the testing are available under \AllDataset\TestingOnly\.
There are two Python codes for frequency health index and angle health index.

### Power Flow Snapshort

### Detailed Testing Procedure
#### Frequency Health Index

## Contributing
Thank you for your interest in improving this work! If you have ideas or suggestions:

Fork the repository and experiment with the code.
Submit a pull request if you'd like to share your updates or fixes.
I appreciate your effort in making this task better, no matter how small the contribution.

## License
You are free to use this code without any restrictions. However, if you utilize the training dataset, please cite this repository in your work.

## Acknowledgments
This code was developed with sponsorship from the Electric Power Research Institute under agreement 10015026.

## Contact
For further contact, email kyamashi@ucr.edu and koji.yamashita.jp@ieee.org
