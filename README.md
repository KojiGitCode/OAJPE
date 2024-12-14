# Machine Learning Datasets and Models for Grid Health Index Prediction Using Graph Attention (GAT) Models
![Single Line Diagram](images/IEEE118-Diagram.png)

## Table of Contents
1. [Key Features](#key-features)
2. [Repository Overview](#repository-overview)
3. [Getting Started](#getting-started)
    - [Prerequisites](#1-Prerequisites)
4. [Results and Reproducibility](#results-and-reproducibility)  
    - [Power Flow Snapshot](#1-power-flow-snapshot)  
    - [Detailed Testing Procedure](#2-detailed-testing-procedure)
    - [Dynamic Simulation](#3-dynamic-simulation) 
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)
 
---

## Key Features
This repository includes:
- Training and testing datasets
- 25,225 power flow snapshots [^1]
- Dynamic simulation results were generated at 40% load with both even and uneven generation dispatches, and at 100% load with no dispatch adjustment required.[^2]
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
There are two Python codes for the frequency health index and angle health index.  

### Power Flow Snapshot  
- The impedance data is available [here](AllDataset/PowerFlowSnapshot/SystemData.txt). Note that all data are based on per unit with a 100 MVA base.  
- Power flow results are available [here](AllDataset/PowerFlowSnapshot/). 

### Detailed Testing Procedure
#### Frequency Health Index

Here are the directory trees.

<pre style="line-height: 1;">
    GAT500b.py
    Sample_Datasets/
    ├── Freq/
        ├── PMUmissLD.lst
        ├── PMUmissSC.lst
        ├── PMUmissSG.lst
        ├── PMUmissSW.lst
        ├── PMUmissTT.lst
        ├── PMUmissTTS.lst
        ├── branch_feature_default.lst
        ├── node_feature_default.lst
        ├── F1index/
            ├── ENN32BNN64GATH4LR0.0001Enc1A.pth    
            ├── ENN32BNN64GATH4LR0.0001Enc1A.enc    
        ├── TestingDataset/
            ├── AllFeatire_GCN_DPG.nod.csv
            ├── AllFeatire_GCN_DPL.nod.csv
            ...
            ├── AllFeatire_GCN_VM.node.csv
            ├── EdgeIndex.csv
            ├── Labels_GCN_Freq.inx.csv
            ├── N-1FullList.txt
</pre>

- The Python code is located at [here](AllDataset/TestingOnly/FrequencyHealthIndex/GAT500b.py).
- The off-the-shelf model is stored at [here](AllDataset/FrequencyHealthIndex/). A file with a *.pth* extension denotes the model, while a file with a *.enc* denotes a flag file for the use of the ordinal encoder.
- Files with a *.lst* extension are stored at [here](AllDataset/TestingOnly/ListFiles/).
- Files with a *.csv* extension are compressed as a 7z-file locatted at [here](AllDataset/TestingOnly/FrequencyHealthIndex/).

Please deploy these files referring to the above directory trees.

The Python code allows users to specify:

- Imputation methods (using `--imp`): 0 for zero imputation, 1 for peak value replacement, 2 for pseudo-PMU measurement.
- Ordinal encoder usage (using `--enc`): `Y` for with ordinal encoder, `N` for without ordinal encoder.
- Error percentage when the pseudo-PMU measurement method is employed (using `--per`): a range between 0 and 100.


These settings can be reflected using the argument of the Python code when running.
For example, when selecting the pseudo-PMU measurement method with the error percentage of 1\%, please type:

```python GAT500.py --imp 2 --mdl GAT --enc Y --per 1```

Currently, only GAT models are available on \AllDatset\OffTheShelfModel. *"--mdl GAT"* is typed for using the graph attention network model. 

Please ensure the order of these arguments. The code does not work when the order is different.

#### Angle Health Index (Between Buses 10 and 12)

Here are the directory trees.

<pre style="line-height: 1;">
    GAT800a.py
    Sample_Datasets/
    ├── Angl/
        ├── PMUmissLD.lst
        ├── PMUmissSC.lst
        ├── PMUmissSG.lst
        ├── PMUmissSW.lst
        ├── PMUmissTT.lst
        ├── PMUmissTTS.lst
        ├── branch_feature_default.lst
        ├── node_feature_default.lst
        ├── F1index/
        ├── ENN32BNN64GAT3H2LR.0001Enc1.pth    
        ├── ENN32BNN64GAT3H2LR.0001Enc0.enc    
        ├── TestingDataset2/
            ├── AllFeatire_GCN_DPG.nod.csv
            ├── AllFeatire_GCN_DPL.nod.csv
            ...
            ├── AllFeatire_GCN_VM.node.csv
            ├── EdgeIndex.csv
            ├── Labels_GCN_Freq.inx.csv
            ├── N-1FullList.txt
</pre>

- The Python code is located at [here](AllDataset/TestingOnly/AngleHealthIndex/GAT500b.py).
- The off-the-shelf model is stored at [here](AllDataset/AngleHealthIndex/). A file with a *.pth* extension denotes the model, while a file with a *.enc* denotes a flag file for the use of the ordinal encoder.
- Files with a *.lst* extension are stored at [here](AllDataset/TestingOnly/ListFiles/).
- Files with a *.csv* extension are compressed as a 7z-file locatted at [here](AllDataset/TestingOnly/AngleHealthIndex/).

Please deploy these files referring to the above directory trees.

The Python code allows users to specify:

- Imputation methods (using `--imp`): 0 for zero imputation, 1 for peak value replacement, 2 for pseudo-PMU measurement.
- Ordinal encoder usage (using `--enc`): `Y` for with ordinal encoder, `N` for without ordinal encoder.
- Error percentage when the pseudo-PMU measurement method is employed (using `--per`): a range between 0 and 100.


These settings can be reflected using the argument of the Python code when running.
For example, when selecting the pseudo-PMU measurement method with the error percentage of 1\%, please type:

```python GAT800a.py --imp 2 --mdl GAT --enc Y --per 1```

Currently, only GAT models are available on \AllDatset\OffTheShelfModel. *"--mdl GAT"* is typed for using the graph attention network model. 

Please ensure the order of these arguments. The code does not work when the order is different.

#### Angle Health Index (Between Buses 10 and 49)

Here are the directory trees.

<pre style="line-height: 1;">
    GAT800b.py
    Sample_Datasets/
    ├── Angl/
        ├── PMUmissLD.lst
        ├── PMUmissSC.lst
        ├── PMUmissSG.lst
        ├── PMUmissSW.lst
        ├── PMUmissTT.lst
        ├── PMUmissTTS.lst
        ├── branch_feature_default.lst
        ├── node_feature_default.lst
        ├── F1index/
        ├── ENN32BNN64GAT4H2LR.0001Enc1.pth    
        ├── ENN32BNN64GAT4H2LR.0001Enc0.enc    
        ├── TestingDataset2/
            ├── AllFeatire_GCN_DPG.nod.csv
            ├── AllFeatire_GCN_DPL.nod.csv
            ...
            ├── AllFeatire_GCN_VM.node.csv
            ├── EdgeIndex.csv
            ├── Labels_GCN_Freq.inx.csv
            ├── N-1FullList.txt
</pre>

- The Python code is located at [here](AllDataset/TestingOnly/AngleHealthIndex/GAT500b.py).
- The off-the-shelf model is stored at [here](AllDataset/AngleHealthIndex/). A file with a *.pth* extension denotes the model, while a file with a *.enc* denotes a flag file for the use of the ordinal encoder.
- Files with a *.lst* extension are stored at [here](AllDataset/TestingOnly/ListFiles/).
- Files with a *.csv* extension are compressed as a 7z-file locatted at [here](AllDataset/TestingOnly/AngleHealthIndex/).

Please deploy these files referring to the above directory trees.

The Python code allows users to specify:

- Imputation methods (using `--imp`): 0 for zero imputation, 1 for peak value replacement, 2 for pseudo-PMU measurement.
- Ordinal encoder usage (using `--enc`): `Y` for with ordinal encoder, `N` for without ordinal encoder.
- Error percentage when the pseudo-PMU measurement method is employed (using `--per`): a range between 0 and 100.


These settings can be reflected using the argument of the Python code when running.
For example, when selecting the pseudo-PMU measurement method with the error percentage of 1\%, please type:

```python GAT800b.py --imp 2 --mdl GAT --enc Y --per 1```

Currently, only GAT models are available on \AllDatset\OffTheShelfModel. *"--mdl GAT"* is typed for using the graph attention network model. 

Please ensure the order of these arguments. The code does not work when the order is different.

### Dynamic Simulation
Dynamic simulation results are available [here](AllDataset/DynamicSimulation/).
There are 9 compressed files in this directory.

- The first three digits correspond to the fault duration with the unit of milliseconds.
- The fourth through sixth letters indicate a type of contingencies, i.e., "N-1" contingencies.
- The eighth through tenth digits display the loading levels, e.g., *040* for 40% loading level.
- The eleventh letter of *u*, if any, means the unbalanced dispatching cases.

Under each compressed file, over 300 subfolders are present.
The subfolder name shows the type of contingencies shown below:

1\. Line Outage Event
- L1S means line trip events
- Subsequent 6 digits indicate the *from* Node and *to* Node, respectively.
- F0.5 means that a fault is applied at 50% of the line.

2\. Transformer Outage Event 
- Trans means the type of outage, i.e., transformer outages.
- Subsequent 4 digits indicate the *from* Node and *to* Node.
- R0.1 means the fault impedance (specifically resistance) of 0.1 per unit at the fault location. Note that the fault is an internal fault, e.g., intertern faults.

3\. Power Plant Outage Event
- Plant means the type of outage, i.e., power plant outages.
- Subsequent letter shows the type of synchronous machines: Synchronous generators (G), Synchronous condensers (S)
- Subsequent 4 letters indicate the type of power plants:
    + AGC: Advanced Gas Combined cycle power plant
    + RORH: Run-off-River Hydropower Plant
    + coal: Coal-fired Power Plant
    + GasT: Gas Turbine Power Plant
    + hydr: Hydropower Plant
    + cond: Synchronous condenser 

Under each folder, 13 time series files are stored with the file extension name "dat."
Each file has a time column and 3-8 node voltage time-series responses.
Headers indicate the node number that corresponds to the single-line diagram in this readme.
For example, VT_107 means the node voltage response at Bus 107.

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
