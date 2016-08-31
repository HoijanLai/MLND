import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env, gm, al):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.cur_key = None
        self.cur_reward = None
        self.risk_fq = [] # how many time in every trial the agent takes risk
        self.silly_fq = [] # how many silly move
        self.Q = {} ## (state, action) : Q_val, action
        self.gamma = gm
        self.alpha = al

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.cur_key = None
        self.cur_reward = None
        self.risk_fq.append(0)
        self.silly_fq.append(0)

    def update(self, t):
        # Gather inputs

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)



        # TODO: Update state
        agent_r, agent_l, agent_f, light = inputs['right'], inputs['left'], inputs['oncoming'], inputs['light']

        right_okay = (light == 'green' or (agent_f != 'left' and agent_l != 'forward'))
        left_okay = (light == 'green' and (agent_f == None or agent_f == 'left'))
        forward_okay = (light == 'green')
        late = (deadline < 0)

        self.state = (late, right_okay, left_okay, forward_okay, self.next_waypoint)

        """deprecated"""
        #self.state = (deadline < 0, inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['right'] is 'forward', inputs['left'])

        # TODO: Select action according to your policy

        # greed-epsilon
        epsilon = (0.1 / np.sqrt(t)) if t != 0 else 0.1
        action = None
        maxQ = 0.0
        if random.random() < epsilon:
            action = random.choice(self.env.valid_actions)
            maxQ = self.get_Qval(self.state, action)
        else:
            action, maxQ = self.max_a_Q(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # update data for statistics
        if reward == -1.0:
            self.risk_fq[-1] += 1
        if reward == -0.5:
            self.silly_fq[-1] += 1

        ## learn how to stop, how to avoid collisions
        # TODO: Learn policy based on state, action, reward
        pre_reward = self.cur_reward
        pre_key = self.cur_key
        self.cur_reward = reward
        self.cur_key = (self.state, action)
        if pre_key is not None:
            self.Q[pre_key] = self.Q[pre_key] + self.alpha*(pre_reward + self.gamma*maxQ - self.Q[pre_key])
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_Qval(self, state, action):
        self.Q.setdefault((state, action), 0.0)
        return self.Q[(state,action)]


    def max_a_Q(self, state):
        """give a state and return (best_action, best_Q_val) based on the policy"""
        a_qs = []
        num_none = 0
        for a in self.env.valid_actions:
            if not self.Q.has_key((state, a)):
                self.Q.setdefault((state, a), 0.0)
                num_none += 1
            a_qs.append((a, self.Q[(state, a)]))
        return max(a_qs, key = lambda x : x[1]) if num_none < 4 else (random.choice(self.env.valid_actions), 0.0)


def run(get_result = False, gm = 0.2, al = 0.5):
    """Run the agent for a finite number of trials."""
    if get_result:
        ## print for GridSearch
        print ("Running trial  for gamma = %.1f, alpha = %.1f" %(gm, al))

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, gm = gm, al = al)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    n_trials = 100
    sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "average silly moves for the last 10 trials: ", np.average(a.silly_fq[-10])
    print "average risky moves for the last 10 trials: ", np.average(a.risk_fq[-10])


    """The Following Code is for GridSearch"""
    if get_result:
        summary = sim.rep.summary()
        rate = sum(summary[-1][-10:])/float(10)
        deadline = sum(summary[-2][-10:])/float(10)
        risk_fq = sum(a.risk_fq[-10:])
        print ("success_rate   for gamma = %.1f, alpha = %.1f is %.2f" %(gm, al, rate))
        print ("final_deadline for gamma = %.1f, alpha = %.1f is %.2f" %(gm, al, deadline))
        print ("risk_frequecy  for gamma = %.1f, alpha = %.1f is %d" %(gm, al, risk_fq))
        print
        return (rate, deadline, risk_fq)

## GridSearching for the best gamma & alpha pair
def GridSearch(file_name = None, n_times = 10):
    metrics = np.zeros((3, n_times, 11, 9))
    for i in range(n_times):
        print "+++++++++++++++++++++++++++++++++++++++"
        print "+       GridSearching N = %02d        +" %(i+1)
        print "+++++++++++++++++++++++++++++++++++++++"
        from pandas import DataFrame as df, Series
        import seaborn as sns; sns.set()

        gms = Series([x * 0.1 for x in range(0, 11)], name = 'gamma')
        als = Series([x * 0.1 for x in range(1, 10)], name = 'alpha')

        for m in range(len(gms)):
            for n in range(len(als)):
                 metrics[0][i][m][n], metrics[1][i][m][n], metrics[2][i][m][n] = run(True, gms[m], als[n])

    ## average , variance to show
    avg_metrics = np.average(metrics, axis = 1)
    var_metrics = np.var(metrics, axis = 1)

    success_df = df(avg_metrics[0], columns = als, index = gms)
    deadline_df = df(avg_metrics[1], columns = als, index = gms)
    risk_df = df(avg_metrics[2], columns = als, index = gms)

    success_df_var = df(var_metrics[0], columns = als, index = gms)
    deadline_df_var = df(var_metrics[1], columns = als, index = gms)
    risk_df_var = df(var_metrics[2], columns = als, index = gms)

    # plot heatmap to visualize param and results
    sns.plt.figure(figsize=(36,12))
    sns.plt.subplot(2, 3, 1).set_title("Success Rate")
    ax_rate = sns.heatmap(success_df, annot = True, fmt = ".2f")

    sns.plt.subplot(2, 3, 2).set_title("Final Deadline")
    ax_dl = sns.heatmap(deadline_df, annot = True, fmt = ".2f")

    sns.plt.subplot(2, 3, 3).set_title("Risky Actions")
    ax_n_r = sns.heatmap(risk_df, annot = True, fmt = ".0f")

    sns.plt.subplot(2, 3, 4).set_title("Success Rate Var")
    ax_rate_var = sns.heatmap(success_df_var, annot = True, fmt = ".2f")

    sns.plt.subplot(2, 3, 5).set_title("Final Deadline Var")
    ax_dl_var = sns.heatmap(deadline_df_var, annot = True, fmt = ".2f")

    sns.plt.subplot(2, 3, 6).set_title("Risky Actions Var")
    ax_n_r_var = sns.heatmap(risk_df_var, annot = True, fmt = ".2f")

    sns.plt.savefig(file_name,) if file_name is not None else sns.plt.show()



if __name__ == '__main__':
    run()
    #GridSearch(n_times = 10)
