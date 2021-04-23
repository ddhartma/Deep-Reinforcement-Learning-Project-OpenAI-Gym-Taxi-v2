import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

            INPUTS:
            ------------
            nA - (int) number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):
        """ Obtains the action probabilities corresponding to epsilon-greedy policy 

            INPUTS:
            ------------
                Q_s - (one-dimensional numpy array of floats) action value function for all six actions
                i_episode - (int) episode number
                eps - (float or None) if not None epsilon is constant

            OUTPUTS:
            ------------
                policy_s - (one-dimensional numpy array of floats) probability for all four actions, to get the most likely action

        """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s
    
    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """ Updates the action-value function estimate using the most recent time step 

            INPUTS:
            ------------
                Qsa - (float) action-value function for s_t, a_t
                Qsa_next - (float) action-value function for s_t+1, a_t+1
                reward - (int) reward for t+1
                alpha - (float) step-size parameter for the update step (constant alpha concept)
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1)

            OUTPUTS:
            ------------
                Qsa_update (float) updated action-value function for s_t, a_t
        """
        Qsa_update = Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

        return Qsa_update


    def select_action(self, state, i_episode):
        """ Given the state, select an action.

            INPUTS:
            ------------
                state - (int) the current state of the environment (1...500)

            OUTPUTS:
            ------------
                action - (int) compatible with the task's action space (1...6)
        """
        
        # get epsilon-greedy action probabilities
        policy_s = self.epsilon_greedy_probs(self.Q[state], i_episode)
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        #print(state, action, policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

            INPUTS:
            ------------
                state - (int) the previous state of the environment
                action - (int) the agent's previous choice of action
                reward - (int) last reward received
                next_state - (int) the current state of the environment
                done - (bool) whether the episode is complete (True or False)
        """
        
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward, alpha=0.1, gamma=1)
        
        
    