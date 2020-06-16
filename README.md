
## Overview

A stock trader powered with deep q-network. 


### Dependencies

Python 3. To install all the libraries, run `pip3 install -r requirements.txt`


### Table of content

* `agent.py`: a Dueling Double Deep Q learning agent
* `envs.py`: a simple trading environment for single currency pair, commodity, or stock
* `model.py`: a multi-layer perceptron as the function approximator
* `utils.py`: some utility functions
* `run.py`: train/test logic
* `requirement.txt`: all dependencies
* `data/`: stock price data

### How to run

**To train a Deep Q agent**, run `python3 run.py --mode train --symbol <OHLC csv>`. There are other parameters and I encourage you look at the `run.py` script. After training, a trained model as well as the portfolio value history at episode end would be saved to disk.

**To test the model performance on validation set**, run `python3 run.py --mode validate --symbol <OHLC csv> --weights <trained_model> --scaler <scaler pkl>`, where `<trained_model>` points to the local model weights file. Validation data portfolio value history at episode end would be saved to disk.


**To test the model performance on test set**, run `python3 run.py --mode test --symbol <OHLC csv> --weights <trained_model> --scaler <scaler pkl>`, where `<trained_model>` points to the local model weights file. Test data portfolio value history at episode end would be saved to disk.


### Visualize

**To visualize performance** run `python3 visualize.py --file <pickle_file>`, where `<pickle_file>` points to the local portfolio value history file for a specific train/validate/test file.  



Credits for starter code go to [ShuaiW](https://github.com/ShuaiW/teach-machine-to-trade). 

