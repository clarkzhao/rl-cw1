import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np

class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.current_reward = 0
        self.horizon = 8
        self.current_state_grid = np.zeros([self.horizon,8],'int8')
        self.next_state_grid = np.zeros([self.horizon,8],'int8')
        self.actions = [Action.ACCELERATE,
                        Action.LEFT,
                        Action.RIGHT,
                        Action.BREAK]
        self.current_action = Action.NOOP
        self.q_vals = {}
        # probability for exploration
        self.EPSILON = 0.05
        # step size
        self.ALPHA = 0.1
        # gamma for Q-Learning
        self.GAMMA = 0.9
	self.action_counter = [0,0,0,0]

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        # cv2.imshow("Enduro", self._image)
        # cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        self.current_state_grid = grid[:self.horizon,1:9]
	print self.action_counter
        with open('action34' +str(self.horizon)+'_0050109.csv','a') as f_act:
            np.savetxt(f_act, [self.action_counter], fmt = '%i', delimiter=",")
	self.action_counter = [0,0,0,0]
    def stateToString(self, state):
        return ''.join(str(x) for x in state.reshape(-1).tolist())

    def getQvals(self, state, action):
        key = (self.stateToString(state), action)
        if key in self.q_vals:
            return self.q_vals[key]
        else:
            self.q_vals[key] = 0.0
            return 0.0

    def argMaxQvals(self, state):
        # Get the action that argmax given current state grid
        possible_Qvals = {}
        for action in self.actions:
            possible_Qvals[action] = self.getQvals(state, action)
        maximals = []
        for key, value in possible_Qvals.items():
            if value == possible_Qvals[max(possible_Qvals, key=possible_Qvals.get)]:
                maximals.append(key)
        return maximals[np.random.choice(len(maximals))]

    def maxQvals(self, state):
        # Get the max Q-values from all possible actions
        values = []
        for action in self.actions:
            values.append(self.getQvals(state,action))
        return max(values)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)

        #exploration
        if np.random.binomial(1, self.EPSILON) == 1:
            self.current_action = np.random.choice(self.actions)
        #exploition
        else:
            self.current_action = self.argMaxQvals(self.current_state_grid)

        #Take action and get reward for current action and state
        self.current_reward = self.move(self.current_action)
        self.total_reward += self.current_reward
	
	act_index = self.actions.index(self.current_action)
	self.action_counter[act_index] +=1

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        self.next_state_grid = grid[:self.horizon,1:9]

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        current_q = self.getQvals(self.current_state_grid,self.current_action)
        # print ('The previous q value is {0}').format(current_q)
        new_q = current_q + self.ALPHA * (self.current_reward +
            self.GAMMA * self.maxQvals(self.next_state_grid) - current_q)
        current_key = (self.stateToString(self.current_state_grid),self.current_action)
        # print ('The new q value is {0}').format(new_q)
        self.q_vals[current_key] = new_q
        self.current_state_grid = self.next_state_grid

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        results = []
        results.append([episode, iteration, self.total_reward])
        with open('q34' +str(self.horizon)+'_0050109.csv','a') as f_handle:
            np.savetxt(f_handle, results, fmt = '%i', delimiter=",")
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    a = QAgent()
    with open('q34' +str(a.horizon)+'_0050109.csv', 'w'):
        pass
    with open('action34' +str(a.horizon)+'_0050109.csv','w') as f_act:
	pass
    a.run(True, episodes=100, draw=True)
    print 'Total reward: ' + str(a.total_reward)
