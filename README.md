
## Overview

This is the code for [this](https://youtu.be/rRssY6FrTvU) video on Youtube by Siraj Raval on Q Learning for Trading as part of the Move 37 course at [School of AI](https://www.theschool.ai). Credits for this code go to [ShuaiW](https://github.com/ShuaiW/teach-machine-to-trade). 

Related post: [Teach Machine to Trade](https://shuaiw.github.io/2018/02/11/teach-machine-to-trade.html)

### Dependencies

Python 3. To install all the libraries, run `pip3 install -r requirements.txt`


### Table of content

* `agent.py`: a Deep Q learning agent
* `envs.py`: a simple x-stock trading environment
* `model.py`: a multi-layer perceptron as the function approximator
* `utils.py`: some utility functions
* `run.py`: train/test logic
* `requirement.txt`: all dependencies
* `data/`: stock price data

### How to run

**To train a Deep Q agent**, run `python run.py --mode train`. There are other parameters and I encourage you look at the `run.py` script. After training, a trained model as well as the portfolio value history at episode end would be saved to disk.

**To test the model performance on validation set**, run `python run.py --mode validate --weights <trained_model>`, where `<trained_model>` points to the local model weights file. Test data portfolio value history at episode end would be saved to disk.


**To test the model performance on test set**, run `python run.py --mode test --weights <trained_model>`, where `<trained_model>` points to the local model weights file. Test data portfolio value history at episode end would be saved to disk.







