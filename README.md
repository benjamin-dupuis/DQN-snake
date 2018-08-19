# DQN-snake
TensorFlow implementation of a DQN algorithm to learn to play the game of Snake.
The game was written using Pygame. During training, a Tensorboad file is produced to visualize the performance of the model.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/41129002/44306716-6ddb2100-a362-11e8-8fde-26d8fc375742.gif)


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
git clone https://github.com/benjamin-dupuis/DQN-snake.git
```

This will create a folder on your computer that you can access from the command prompt by executing 

```
cd DQN-snake
```

To start training a new model or to continue training an existing model, run
```
python train.py
```

To test the performance of your gamebot, run 
```
python test.py
```

To play the game yourself, run 
```
python play.py
```

To visualize the performance of the model during training (for example 'new_model') with Tensorboard, open a new cmd prompt, and run

```
tensorboard --logdir DQN-snake/tf_logs/new_model
```


## Results


### Model 1 : 
During training, the raw pixel values are extracted from the game. Then, those values are converted into features by a Convolutional Neural Network (CNN). The chosen architecture was composed of two convolutional layers followed by two fully connected layers. Below is a figure representing the architecture :  

![architecture](assets/architecture_2.PNG)


For the first try, the chosen hyperparameters were the following : 

- memory_size : 50 000
- momentum : 0.95
- discount_rate : 0.90
- training_interval : 2
- learning_rate : 0.0001
- eps_min : 0.1
- eps_decay_steps : 2 000 000
- n_steps : 5 000 000


After 5 000 000 training steps, I tested the gamebot by making it play 200 games. It obtained the following results : 

- Max score : 17.00, Mean score : 2.63, Standard deviation : 3.41. 

The low mean score is representative of the fact that the snake was prone to getting stuck (always repeating the same movements).
In the future, I will try different versions of the model, especially by changing the discount rate. 


## References 

- [deep-q-snake](https://github.com/danielegrattarola/deep-q-snake)
- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [Reinforcement learning](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb)

