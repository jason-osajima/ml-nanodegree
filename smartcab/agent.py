import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QTable(object):
    """A table that an agent uses to look up q-values for different state/action pairs."""
    def __init__(self):
        self.Q_table = {}

    def set(self, state, action, q):
        #sets a new q_value for a given state and action pair.
        k = (state, action)
        self.Q_table[k] = q

    def get(self, state, action):
        #gets a q_value for a corresponding state and action pair.
        k = (state, action)
        return self.Q_table[k, None] 


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = Environment.valid_actions
        self.alpha = 0.5
        self.gamma = 0.8
        #initialize the Q_table
        self.Q_table = QTable()


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_state = None
        self.next_waypoint = None

    def choose_action(self, state):
        q = [self.Q_table.get(state, a) for a in self.actions]
        maxQ = max(q)
        action = self.actions[maxQ]
        return action

    def update_q_table(self, state, action, reward):
        new_waypoint = self.planner.next_waypoint()
        new_inputs = self.env.sense(self)
        new_state = (new_inputs['light'], new_inputs['oncoming'], new_inputs['left'], new_waypoint)
        new_action = self.choose_action(new_state)
        current_q = self.Q_table.get(state, action)

        #Use q learning equation to update q value for given state, action pair for the q_table.
        new_q = current_q + self.alpha * (reward + self.gamma * (max(self.Q_table.get(new_state, new_action))) - current_q)
        
        #set the new_q in the q_table for the given state and action.
        self.Q_table.set(state, action, new_q)



    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        action = choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_q_table(self.state, action, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line




if __name__ == '__main__':
    run()
