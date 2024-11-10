# Vision Network

This project consists in training a CNN,based on the Gatenet or Dronet architecture, to detect drone racing gates and get gate size and location. For more information, consult the [Nano Drone Racing](https://github.com/fed12345/nano-drone-racing) repository.


## Description
A streamlined CNN, based on the Gatenet architecture, has been adapted to minimize computational demand. This network is successfully deployed on a GAP8 processor, achieving a processing rate of 16Hz. The CNN provides data on gate size and location, which serves as input for the positioning algorithm

This repository can train 2 different CNN architetures GateNet and Dronet. As well as an experimental active vision network. Moreover, we can finetune a network by lock in place certain layers and augments training data. 

## Getting Started

### Dependencies

- [Conda](https://www.anaconda.com/download) or Pip

### Installing

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fed12345/visionnet
   cd visionnet
   ```
   

2. **Set Up a Virtual Environment (recommended)**

It’s a best practice to use a virtual environment to manage dependencies. To create a virtual environment, run the following command with conda installed:

```bash
conda create --name visionnet
conda activate visionnet
```

3. **Intall Dependencies**

With the environment active, install all necessary packages by running:

```bash
pip install -r requirements.txt
```

4. **Training data**

The datasets used of training are found in [Gatenet](https://github.com/open-airlab/GateNet)

##  Executing Progam

### Train network

To choose the architecture and train the network:

```bash
python3 train.py -c config/gatenet.json
```

### Data Augmentation

To finetune the model:
```bash
python3 finetune.py 
```

