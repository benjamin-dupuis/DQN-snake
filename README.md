# DQN-snake
TensorFlow implementation of a DQN algorithm to learn to play the game of Snake.
The game was written using Pygame. During training, a Tensorboad file is produced to visualize the performance of the model.


## Requirements

The langage that is used is Python (version 3.5), which can be downloaded at https://www.python.org/. 
The libraries that are needed are the following, which can be downloaded with the pip install command : 

1. TensorFlow 1.5.0 (GPU support recommanded)
2. pygame 
3. numpy

Detailed and clear explanations on how to download the GPU version of TensorFlow on Windows 10 can be found at http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html .


For other platforms, explanations can be found at
https://www.tensorflow.org/install . 



## Usage

To download the repository, open a cmd prompt and execute 
```
git clone https://github.com/BenjaminCanada/DQN-snake.git
```

This will create a folder on your computer that you can access from the command prompt by executing 

```
cd DQN-snake
```

To start training a new model or to continue training an existing model, run
```
python train.py
```

To test a model, run 
```
python test.py
```

To play the game yourself, run 
```
python play.py
```

To visualize the performance of the model during training (for example 'new_model') with Tensorboard, open a new cmd prompt, and run

```
tensorboard --logidr DQN-snake/tf_logs/new_model
```


## Results


### Model 1 : 
During training, the raw pixel values are extracted from the game. Then, those values are converted into features by a Convolutional Neural Network (CNN). The chosen architecture was composed of two convolutional layers followed by two fully connected layers. Below is a schema representing the architecture :  

![architecture](assets/architecture_2.PNG)


For our first try, the chosen hyperparameters were the following : 

- memory size : 10 000
- momentum = 0.95
- discount rate = 0.90
- training_interval = 2
- learning rate = 0.0001


## References 

- [deep-q-snake](https://github.com/danielegrattarola/deep-q-snake)
- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [Reinforcement learning](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb)

