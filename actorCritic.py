import numpy as np
import tensorflow as tf
from utils import ReplayMemory
import os


INPUT_HEIGHT = 84     
INPUT_WIDTH = 84     
CHANNELS = 4       
N_OUTPUTS = 4  # Number of possible actions that the agent can make (the four directions)


models_path = './models/new_model/'

# Create the folder where we will save our model if it does not already exist
if not os.path.isdir(models_path):
    os.makedirs(models_path)

checkpoint_path = models_path + 'dqn.ckpt'


# To be more robust to outliers, we use a quadratic loss for small errors, and a linear loss for large ones.
def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class ActorCritic:

    def __init__(self, sess, learning_rate=0.0001, momentum=0.95, memory_size=10000, discount_rate=0.90):
        self.activation = tf.nn.relu
        self.optimizer = tf.train.MomentumOptimizer
        self.learning_rate = learning_rate
        self.momentum = momentum

        self._build_graph()

        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)

        '''
        The discount rate is the parameter that indicates how many actions will be considered in the future to evaluate
        the reward of a given action.
        A value of 0 means the agent only considers the present action, and a value close to 1 means the agent
        considers actions very far in the future.
        '''
        self.discount_rate = discount_rate

        self.eps_min = 0.1
        self.eps_decay_steps = 2000000

        self.sess = sess
        self.init = tf.global_variables_initializer()
        self.y_vals = None

    def cnn_model(self, X_state, name):
        """
        Creates a CNN network with three convolutional layers followed by two fully connected layers.
        
        :param X_state: Placeholder for the state of the game
        :param name: Name of the network (actor or critic)
        :return : The output (logits) layer and the trainable variables
        """

        initializer = tf.contrib.layers.variance_scaling_initializer()

        conv1_fmaps = 32
        conv1_ksize = 8
        conv1_stride = 4
        conv1_pad = 'SAME'

        conv2_fmaps = 64
        conv2_ksize = 4
        conv2_stride = 2
        conv2_pad = 'SAME'

        n_fc1 = 256

        with tf.variable_scope(name) as scope:

            conv1 = tf.layers.conv2d(X_state, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                     activation=self.activation, strides=conv1_stride, padding=conv1_pad, name='conv1')

            conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                                     activation=self.activation, strides=conv2_stride, padding=conv2_pad, name='conv2')

            conv2_flat = tf.reshape(conv2, shape=[-1, conv2_fmaps*11*11])

            fc1 = tf.layers.dense(conv2_flat, n_fc1, activation=self.activation, name='fc1', kernel_initializer=initializer)

            logits = tf.layers.dense(fc1, N_OUTPUTS, kernel_initializer=initializer)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
        return logits, trainable_vars_by_name

    def _build_graph(self):
        """
        Creates the Tensorflow graph of the CNN network.
        Two networks will be used, one for the actor, and one for the critic.
        """

        X_state = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH, CHANNELS])
        actor_q_values, actor_vars = self.cnn_model(X_state, name="actor")   
        critic_q_values, critic_vars = self.cnn_model(X_state, name="critic")   

        with tf.variable_scope("train"):
            X_action = tf.placeholder(tf.int32, shape=[None])
            y = tf.placeholder(tf.float32, shape=[None, 1])

            '''A one hot vector (tf.one_hot) is used to only keep the Q-value corresponding to chosen action in the memory. 
            By multiplying the one-hot vector with the actor_q_values, this will zero out all of the Q-values except
            for the one corresponding to the memorized action. Then, by making sum along the first axis (axis=1), 
            we obtain the desired Q-value prediction for each memory.
            '''
            q_value = tf.reduce_sum(actor_q_values * tf.one_hot(X_action, N_OUTPUTS),
                                    axis=1, keep_dims=True)
            error = tf.abs(y - q_value)
            loss = tf.reduce_mean(clipped_error(error))
            global_step = tf.Variable(0, trainable=False, name='global_step')  # iteration step
            optimizer = self.optimizer(self.learning_rate, self.momentum, use_nesterov=True)
            training_op = optimizer.minimize(loss, global_step=global_step)

        self.saver = tf.train.Saver()
        self.X_state = X_state
        self.X_action = X_action
        self.y = y
        self.training_op = training_op
        self.loss = loss
        self.actor_q_values, self.actor_vars = actor_q_values, actor_vars
        self.critic_q_values, self.critic_vars = critic_q_values, critic_vars
        self.global_step = global_step

        with tf.variable_scope('summary'):
            self.loss_summary = tf.summary.scalar('loss', loss)
            self.mean_score = tf.placeholder(tf.float32, None)    
            self.score_summary = tf.summary.scalar('mean score', self.mean_score)
            self.summary_merged = tf.summary.merge([self.loss_summary, self.score_summary])

    def start(self):
        """
        Intialize the model or restore the model if it already exists.
        
        :return: Iteration that we want the model to start training
        """
        if os.path.isfile(checkpoint_path + '.index'):
            self.saver.restore(self.sess, checkpoint_path)
            training_start = 1  
            print('Restoring model...')
        else:
            # Make the model warm up before training
            training_start = 10000  
            self.init.run()
            self.make_copy().run()
            print('New model...')
        return training_start

    def train(self, file_writer, mean_score):
        """
        Trains the agent and writes regularly a training summary.
        
        :param file_writer: file where the training summary will be written for Tensorboard visualization
        :param mean_score: mean game score
        """
        copy_steps = 5000  
        save_steps = 2000   
        summary_steps = 500 

        cur_states, actions, rewards, next_states, dones = self.sample_memories()

        next_q_values = self.critic_q_values.eval(feed_dict={self.X_state: next_states})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_vals = rewards + (1-dones)*self.discount_rate * max_next_q_values
        _, loss_val = self.sess.run([self.training_op, self.loss],
                                    feed_dict={self.X_state: cur_states, self.X_action: actions, self.y: y_vals})

        step = self.global_step.eval()

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            self.make_copy().run()

        # Save the model regularly
        if step % save_steps == 0:
            self.saver.save(self.sess, checkpoint_path)

        # Write the training summary regularly
        if step % summary_steps == 0:
            summary = self.sess.run(self.summary_merged, feed_dict={
                self.X_state: cur_states, self.X_action: actions, self.y: y_vals, self.mean_score: mean_score})

            file_writer.add_summary(summary, step)

    def predict(self, cur_state):
        """
        Makes the actor predict q-values based on the current state of the game.
        
        :param cur_state: Current state of the game
        :return The Q-values predicted by the actor
        """
        q_values = self.actor_q_values.eval(feed_dict={self.X_state: [cur_state]})
        return q_values

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def act(self, cur_state, step):
        """
        :param cur_state: Current state of the game
        :param step: Training step
        :return: Action selected by the agent
        """
        q_values = self.predict(cur_state)
        eps_max = 1.0
        epsilon = max(self.eps_min, eps_max - (eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(N_OUTPUTS)  # Random action
        else:
            return np.argmax(q_values)  # Optimal action

    def make_copy(self):
        """
        Makes regular copies of the training varibales from the critic to the actor.
        Credits goes to https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb.
        
        :return: A copy of the training variables
        """
        copy_ops = [target_var.assign(self.actor_vars[var_name])
                    for var_name, target_var in self.critic_vars.items()]
        copy_online_to_target = tf.group(*copy_ops)
        return copy_online_to_target

    def sample_memories(self, batch_size=32):
        """
        Extracts memories from the agent's memory.
        Credits goes to https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb.
        
        :param batch_size: Size of the batch that we extract form the memory
        :return: State, action, reward, next_state, and done values as np.arrays
        """
        cols = [[], [], [], [], []]  # state, action, reward, next_state, done
        for memory in self.memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
