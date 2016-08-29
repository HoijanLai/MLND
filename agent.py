import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    gamma = 0.6
    alpha = 0.1

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.cur_key = None
        self.cur_reward = None
        self.Q = {} ## (state, action) : Q_val, action

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.cur_key = None
        self.cur_reward = None

    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)



        # TODO: Update state
        ## deadline excess, traffic light, to-go direction, intentions of the others
        self.state = (deadline < 0, inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['right'], inputs['left'])

        # TODO: Select action according to your policy
        a_qs = []
        num_none = 0
        for a in [None, 'left', 'right', 'forward']:
            if not self.Q.has_key((self.state, a)):
                self.Q.setdefault((self.state, a), 0.0)
                num_none += 1
            a_qs.append((a, self.Q[(self.state, a)]))
        a_q_max = max(a_qs, key = lambda x : x[1])

        action = a_q_max[0] if num_none < 4 else self.next_waypoint

        # Execute action and get reward
        reward = self.env.act(self, action)

        ## learn how to stop, how to avoid collisions
        # TODO: Learn policy based on state, action, reward
        pre_reward = self.cur_reward
        pre_key = self.cur_key
        self.cur_reward = reward
        self.cur_key = (self.state, action)
        if pre_key is not None:
            self.Q[pre_key] = self.Q[pre_key] + self.alpha*(pre_reward + self.gamma*a_q_max[1] - self.Q[pre_key])

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=600)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
