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
python train.py --modelName <nameOfYourModel>
```

Arguments can be passed in the previous command to try differents training parameters : 

<table>
  <tr>
    <th>Option</th>
    <th>Description</th>
    <th>Default value</th>
    <th>Required</th>
  </tr>
  <tr>
    <td>--modelName</td>
    <td>Name of the model.<br><br>Example : --modelName new_model</td>
    <td>---</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>--learningRate</td>
    <td>Rate at which the agent is learning.<br><br>Example : --learningRate 0.002</td>
    <td>.0001</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--memorySize</td>
    <td>Number of events remembered by the agent.<br><br>Example : --memorySize 50000</td>
    <td>100000</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--discountRate</td>
    <td>The discount rate is the<span style="font-weight:bold"> </span>parameter that indicates<br>how<span style="font-weight:bold"> </span>many actions will be considered in the future <br>to evaluate the reward of a given action.  <br>A value of 0 means the agent only <br>considers the present action,<br>and a value close to 1 means the agent<br>considers actions very far in the future.<br><br>Example : --discountRate 0.99</td>
    <td>0.95</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--epsilonMin</td>
    <td>Percentage of random actions selected by the agent.<br><br>Example: --epsilonMin 0.10</td>
    <td>0.05</td>
    <td>No</td>
  </tr>
</table>





To test the performance of your gamebot, run 
```
python test.py --modelName <nameOfYourModel>
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


### Model 2:

For the second experiment, I kept the same CNN architecture, but chose the following parameters:


- memory_size : 100 000
- momentum : 0.95
- discount_rate : 0.95
- training_interval : 2
- learning_rate : 0.0001
- eps_min : 0.05
- eps_decay_steps : 2 000 000
- n_steps : 5 000 000


Like reviously, I made the gamebot play 200 games, and obtained the following results:

Max score: 16.00, Mean score: 3.35, Standard deviation: 3.79.

The mean score improved, but more ajustements are necessary. A way to improve the results would be to give the apple the same initial position for each game, which is random at the moment. That would enable the snake to get an easier start and prevent it of being stuck at the beginning of a game.


### Model 3:

For the third experiment, I kept the same parameters as in model 2, but I fixed the initial position of the apple. That made the snake have an easier start, and therefore the number of times it got stuck reduced. When testing, the numbers after the 200 games played improved : 

Max score: 15.00, Mean score: 4.25, Standard deviation: 3.29.


## References 

- [deep-q-snake](https://github.com/danielegrattarola/deep-q-snake)
- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [Reinforcement learning](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb)

