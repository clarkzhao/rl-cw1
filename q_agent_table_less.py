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
        self.current_state_grid = np.zeros([6,])
        self.next_state_grid = np.zeros([6,])
        self.actions = {0:Action.ACCELERATE,
                        1:Action.LEFT,
                        2:Action.RIGHT,
                        3:Action.BREAK}
        self.current_action = 0
        # probability for exploration
        self.EPSILON = 0.05
        # step size
        self.ALPHA = 0.01
        # gamma for Q-Learning
        self.GAMMA = 1
        # State: {sensor 0, 90,180, deviation}
        self.q_table = np.zeros([2,2,2,10,4])
        # self.action_counter = [0,0,0,0]

    def getState(self, grid):
        agent_pt = np.argmax(grid)
        sensor_0 = agent_pt
        for i in range(agent_pt):
            if grid[0,i] == 1:
                sensor_0 -= i
                break

        if sensor_0 > 2:
            sensor_0 = 1
        else:
            sensor_0 = 0

        sensor_180 = 9-agent_pt
        for i in range(9-agent_pt):
            if grid[0,i+agent_pt] ==1:
                sensor_180 = i
                break
        if sensor_180 > 2:
            sensor_180 = 1
        else:
            sensor_180 = 0

        sensor_90 = 10
        sensor_90_l = 10
        sensor_90_r = 10

        if agent_pt <= 1:
            sensor_90_l = 10
        else:
            for i in range(10):
                if grid[i, agent_pt-1] == 1:
                    sensor_90_l = i
                    break

        if agent_pt >= 8:
            sensor_90_r = 10
        else:
            for i in range(10):
                if grid[i, agent_pt+1] == 1:
                    sensor_90_r = i
                    break

        for i in range(10):
            if grid[i, agent_pt] == 1:
                sensor_90 = i
                break

        if sensor_90 > 5:
            sensor_90 = 1
        else:
            sensor_90 = 0

        if sensor_90_l > 5:
            sensor_90_l = 1
        else:
            sensor_90_l = 0

        if sensor_90_r > 5:
            sensor_90_r = 1
        else:
            sensor_90_r =0

        sensor_90_all = sensor_90 * sensor_90_l * sensor_90_r
        deviation = 4-agent_pt
        return np.array([sensor_0, sensor_90_all, sensor_180, deviation])

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        # cv2.imshow("Enduro", self._image)
        # cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        # self.current_state_grid = grid[:self.horizon,1:1+self.vertical]
        self.current_state_grid = self.getState(grid)
        # with open('action_counter.csv','a') as f_act:
            # np.savetxt(f_act, [self.action_counter], fmt = '%i', delimiter=",")
        # self.action_counter = [0,0,0,0]
    def getQvals(self, state, action):
        return self.q_table[state[0],state[1],state[2],state[3],action]

    def argmax(self, unique=True):
        state_value = self.q_table[self.current_state_grid[0],
                                self.current_state_grid[1],
                                self.current_state_grid[2],
                                self.current_state_grid[3],:]
        maxValue = np.max(state_value)
        candidates = np.where(np.asarray(state_value) == maxValue)[0]
        if unique:
            return np.random.choice(candidates)
        return list(candidates)

    def doAction(self):
        return self.actions[self.current_action]

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
            self.current_action = np.random.randint(4, size=1)[0]
        #exploition
        else:
            self.current_action = self.argmax()
        #Take action and get reward for current action and state
        self.current_reward = self.move(self.doAction())
        self.total_reward += self.current_reward
        # self.action_counter[self.current_action] +=1

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        self.next_state_grid = self.getState(grid)

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        current_q = self.getQvals(self.current_state_grid,self.current_action)
        # print ('The previous q value is {0}').format(current_q)
        new_q = current_q + self.ALPHA * (self.current_reward +
            self.GAMMA * self.maxQvals(self.next_state_grid) - current_q)
        # print ('The new q value is {0}').format(new_q)
        self.q_table[self.current_state_grid[0],
                    self.current_state_grid[1],
                    self.current_state_grid[2],
                    self.current_state_grid[3],
                    self.current_action] = new_q
        self.current_state_grid = self.next_state_grid
    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        results = []
        results.append([episode, iteration, self.total_reward])
        with open('q_table_less_0050011.csv','a') as f_handle:
            np.savetxt(f_handle, results, fmt = '%i', delimiter=",")
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    a = QAgent()
    with open('q_table_less_0050011.csv', 'w'):
        pass
    a.run(True, episodes=100, draw=True)
    print 'Total reward: ' + str(a.total_reward)
