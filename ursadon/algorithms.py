import pandas
import numpy
import tensorflow as tf
import random
import os


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class q_table:

  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    self.actions = actions  # a list
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon = e_greedy
    self.q_table = pandas.DataFrame(columns=self.actions, dtype=numpy.float64)
    self.disallowed_actions = {}

  def choose_action(self, observation, excluded_actions=[]):
    observation = str(observation)

    self.check_state_exist(observation)

    self.disallowed_actions[observation] = excluded_actions

    state_action = self.q_table.ix[observation, :]

    for excluded_action in excluded_actions:
      del state_action[excluded_action]

    if numpy.random.uniform() < self.epsilon:
      # some actions have the same value
      state_action = state_action.reindex(numpy.random.permutation(state_action.index))

      action = state_action.idxmax()
    else:
      action = numpy.random.choice(state_action.index)

    return action

  def learn(self, s, a, r, s_):
    s = str(s)
    s_ = str(s_)

    if s == s_:
      return

    self.check_state_exist(s_)
    self.check_state_exist(s)

    q_predict = self.q_table.ix[s, a]

    s_rewards = self.q_table.ix[s_, :]

    if s_ in self.disallowed_actions:
      for excluded_action in self.disallowed_actions[s_]:
        del s_rewards[excluded_action]

    if s_ != 'terminal':
      q_target = r + self.gamma * s_rewards.max()
    else:
      q_target = r  # next state is terminal

    # update
    self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      # append new state to q table
      self.q_table = self.q_table.append(pandas.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class q_network:

  def __init__(self, actions, state_size, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, load_file=None):
    tf.reset_default_graph()
    self.inputs1 = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
    # self.W = tf.Variable(tf.random_uniform([8, 8], 0, 0.01))
    # self.W = tf.Variable(tf.random_uniform([state_size, len(actions)], -0.0001, 0.0001))
    self.W = tf.Variable(tf.random_uniform([state_size, len(actions)], 0, 0))
    self.Qout = tf.matmul(self.inputs1, self.W)
    self.predict = tf.argmax(self.Qout, 1)

    self.nextQ = tf.placeholder(shape=[1, len(actions)], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    self.updateModel = trainer.minimize(loss)

    self.sess = tf.Session()

    self.saver = tf.train.Saver()

    if load_file and os.path.isfile(load_file + '.ckpt.index'):
      self.saver.restore(self.sess, load_file + '.ckpt')
      print('Learning loaded')
    else:
      # print('File Not Found')
      # exit()
      self.sess.run(tf.global_variables_initializer())

    # self.sess.run(tf.global_variables_initializer())

    self.actions = actions
    """
    self.lr = learning_rate
    """
    self.gamma = reward_decay
    self.epsilon = e_greedy
    """
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    """
    self.disallowed_actions = {}
    self.state_size = state_size

    self.save_file = load_file

  def save(self):
    self.saver.save(self.sess, self.save_file + '.ckpt')

  def choose_action(self, observation, excluded_actions=[]):

    action, allQ = self.sess.run(
      [self.predict, self.Qout],
      # feed_dict={self.inputs1:np.identity(12)[s:s + 1]}
      feed_dict={self.inputs1: numpy.reshape(observation, [1, self.state_size])}
    )

    self.lastQ = allQ

    print(action[0], allQ[0])

    if numpy.random.rand(1) > self.epsilon:
      return random.choice(list(enumerate(allQ[0])))[0]

    return action[0]

    """
    self.check_state_exist(observation)
    
    self.disallowed_actions[observation] = excluded_actions
    
    state_action = self.q_table.ix[observation, :]
    
    for excluded_action in excluded_actions:
        del state_action[excluded_action]
    
    if np.random.uniform() < self.epsilon:
        # some actions have the same value
        state_action = state_action.reindex(np.random.permutation(state_action.index))
        
        action = state_action.idxmax()
    else:
        action = np.random.choice(state_action.index)
        
    return action
    """

  def learn(self, s, a, r, s_):
    Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: numpy.reshape(s_, [1, self.state_size])})
    maxQ1 = numpy.max(Q1)
    # targetQ = self.sess.run(self.Qout, feed_dict={self.inputs1: np.reshape(s, [1, 8])})
    targetQ = self.lastQ
    targetQ[0, a] = r + self.gamma * maxQ1

    _, W1 = self.sess.run([self.updateModel, self.W], feed_dict={self.inputs1: numpy.reshape(s, [1, self.state_size]), self.nextQ: targetQ})

    """
    if s == s_:
        return
    
    self.check_state_exist(s_)
    self.check_state_exist(s)
    
    q_predict = self.q_table.ix[s, a]
    
    s_rewards = self.q_table.ix[s_, :]
    
    if s_ in self.disallowed_actions:
        for excluded_action in self.disallowed_actions[s_]:
            del s_rewards[excluded_action]
    
    if s_ != 'terminal':
        q_target = r + self.gamma * s_rewards.max()
    else:
        q_target = r  # next state is terminal
        
    # update
    self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
    """

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      # append new state to q table
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class dqn:

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            load_file=None
    ):
      self.n_actions = n_actions
      self.n_features = n_features
      self.lr = learning_rate
      self.gamma = reward_decay
      self.epsilon_max = e_greedy
      self.replace_target_iter = replace_target_iter
      self.memory_size = memory_size
      self.batch_size = batch_size
      self.epsilon_increment = e_greedy_increment
      self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

      # total learning step
      self.learn_step_counter = 0

      # initialize zero memory [s, a, r, s_]
      self.memory = numpy.zeros((self.memory_size, n_features * 2 + 2))

      # consist of [target_net, evaluate_net]
      self._build_net()
      t_params = tf.get_collection('target_net_params')
      e_params = tf.get_collection('eval_net_params')
      self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

      self.sess = tf.Session()

      if output_graph:
        # $ tensorboard --logdir=logs
        # tf.train.SummaryWriter soon be deprecated, use following
        tf.summary.FileWriter("logs/", self.sess.graph)

      self.saver = tf.train.Saver()

      if load_file and os.path.isfile(load_file + '.ckpt.index'):
        self.saver.restore(self.sess, load_file + '.ckpt')
        print('Learning loaded')
      else:
        self.sess.run(tf.global_variables_initializer())

      self.cost_his = []

      self.save_file = load_file

    def _build_net(self):
      # ------------------ build evaluate_net ------------------
      self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
      self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
      with tf.variable_scope('eval_net'):
        # c_names(collections_names) are the collections to store variables
        c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

        # first layer. collections is used later when assign to target net
        with tf.variable_scope('l1'):
          w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
          b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
          l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

        # second layer. collections is used later when assign to target net
        with tf.variable_scope('l2'):
          w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
          b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
          self.q_eval = tf.matmul(l1, w2) + b2

      with tf.variable_scope('loss'):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
      with tf.variable_scope('train'):
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

      # ------------------ build target_net ------------------
      self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
      with tf.variable_scope('target_net'):
        # c_names(collections_names) are the collections to store variables
        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

        # first layer. collections is used later when assign to target net
        with tf.variable_scope('l1'):
          w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
          b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
          l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

        # second layer. collections is used later when assign to target net
        with tf.variable_scope('l2'):
          w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
          b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
          self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
      if not hasattr(self, 'memory_counter'):
        self.memory_counter = 0

      transition = numpy.hstack((s, [a, r], s_))

      # replace the old memory with new memory
      index = self.memory_counter % self.memory_size
      self.memory[index, :] = transition

      self.memory_counter += 1

    def choose_action(self, observation, excluded_actions=[]):
      # to have batch dimension when feed into tf placeholder
      observation = observation[numpy.newaxis, :]

      if numpy.random.uniform() < self.epsilon:
        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        print("DQN: ", actions_value)
        action = numpy.argmax(actions_value)
      else:
        action = numpy.random.randint(0, self.n_actions)
      return action

    def learn(self):
      # check to replace target parameters
      if self.learn_step_counter % self.replace_target_iter == 0:
        self.sess.run(self.replace_target_op)
        print('\ntarget_params_replaced\n')

      # sample batch memory from all memory
      if self.memory_counter > self.memory_size:
        sample_index = numpy.random.choice(self.memory_size, size=self.batch_size)
      else:
        sample_index = numpy.random.choice(self.memory_counter, size=self.batch_size)
      batch_memory = self.memory[sample_index, :]

      q_next, q_eval = self.sess.run(
        [self.q_next, self.q_eval],
        feed_dict={
          self.s_: batch_memory[:, -self.n_features:],  # fixed params
          self.s: batch_memory[:, :self.n_features],  # newest params
        })

      # change q_target w.r.t q_eval's action
      q_target = q_eval.copy()

      batch_index = numpy.arange(self.batch_size, dtype=numpy.int32)
      eval_act_index = batch_memory[:, self.n_features].astype(int)
      reward = batch_memory[:, self.n_features + 1]

      q_target[batch_index, eval_act_index] = reward + self.gamma * numpy.max(q_next, axis=1)

      """
      For example in this batch I have 2 samples and 3 actions:
      q_eval =
      [[1, 2, 3],
       [4, 5, 6]]

      q_target = q_eval =
      [[1, 2, 3],
       [4, 5, 6]]

      Then change q_target with the real q_target value w.r.t the q_eval's action.
      For example in:
          sample 0, I took action 0, and the max q_target value is -1;
          sample 1, I took action 2, and the max q_target value is -2:
      q_target =
      [[-1, 2, 3],
       [4, 5, -2]]

      So the (q_target - q_eval) becomes:
      [[(-1)-(1), 0, 0],
       [0, 0, (-2)-(6)]]

      We then backpropagate this error w.r.t the corresponding action to network,
      leave other action as error=0 cause we didn't choose it.
      """

      # train eval network
      _, self.cost = self.sess.run([self._train_op, self.loss],
                                   feed_dict={self.s: batch_memory[:, :self.n_features],
                                              self.q_target: q_target})
      self.cost_his.append(self.cost)

      # increasing epsilon
      self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
      self.learn_step_counter += 1

    def plot_cost(self):
      import matplotlib.pyplot as plt
      plt.plot(numpy.arange(len(self.cost_his)), self.cost_his)
      plt.ylabel('Cost')
      plt.xlabel('training steps')
      plt.show()

    def save(self):
      self.saver.save(self.sess, self.save_file + '.ckpt')
