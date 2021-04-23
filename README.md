[image1]: assets/openai_taxi.png "image1"

# Deep Reinforcement Learning Project - OpenAi Gym Taxi-v2

## Content
- [Introduction](#intro)
- [OpenAI Gym - taxi-v2 - environment](#openai_taxi)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## OpenAI Gym - taxi-v2 - environment <a name="openai_taxi"></a>
- Image of environment

    ![image1]

- 5-by-5 grid world
- There are four specially-designated locations in this world, marked as R(ed), B(lue), G(reen), and Y(ellow).
- Episodic task
- In each episode, the taxi starts in a randomly-chosen square.
- There is a passenger at one of the four locations (chosen randomly), and that passenger wishes to be transported to one of the fourlocations (also chosen randomly).
- The taxi must go to the passenger’s location (the “source”), pickup the passenger, go to the destination location (the “destination”), and put down the passenger there. (To keep things uniform, the taxi must pick up and dropoff the passenger even if he/she is already located at the destination!) The episode ends when the passenger is deposited at the destination location.
- **Actions**: There are six primitive actions in this domain: (a) four navigation actions that move the taxione square North, South, East, or West, (b) a Pickup action, and (c) a Putdown action. 
- Each action is deterministic.  There is a reward of −1 for each action and an additional reward of +20 for successfully delivering the passenger. There is a reward of −10 if the taxi attempts to execute the Putdown or Pickup actions illegally. If a navigation action would cause the taxi to hit a wall, the action is a no-op, and there is only the usual reward of −1.
- We seek a policy that maximizes the total reward per episode. 
- **States**: There are 500 possible states: 25 squares, 5 locations for the passenger (counting the four starting locations and the taxi), and 4 destinations.
- This task has a simple hierarchical structure in which there are two main sub-tasks: Get the passenger and Deliver the passenger.  Each of these subtasks in turn involves the subtask of navigating to one of the four locations and then performing a Pickup or Putdown action.

- To construct a MAXQ decomposition for the taxi problem, we must identify a set of individual subtasks that we believe will be important for solving the overall task. In this case, let us define the following four tasks:
    - **Navigate(t)**: In this subtask, the goal is to move the taxi from its current location to one of the four target locations, which will be indicated by the formal parametert.
    - **Get**: In this subtask, the goal is to move the taxi from its current location to the passenger’s current location and pick up the passenger.
    - **Put**: The goal of this subtask is to move the taxi from the current location to the passenger’s destination location and drop off the passenger.
    - **Root**: This is the whole taxi task.

## Files in the repo
The workspace contains three files:
- **agent.py**: The reinforcement learning agent is developed.
- **monitor.py**: The interact function tests how well the agent learns from interaction with the environment.
- **main.py**: Run this file in the terminal to check the performance of the agent.

## Implementation - agents.py
- Open Python file ```agents.py```
- Implemented method: TD control: Sarsamax (Q-Learning)
    ```
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
    ```

## Implementation - monitor.py
- Open Python file ```monitor.py```
    ```
    from collections import deque
    import sys
    import math
    import numpy as np

    def interact(env, agent, num_episodes=1000, window=100):
        """ Monitor agent's performance.
        
            INPUTS:
            ------------
                env - (instance of OpenAI Gym) Taxi-v1 environment
                agent - (instance of class Agent) see Agent.py for details
                num_episodes - (int) number of episodes of agent-environment interaction
                window - (int) number of episodes to consider when calculating average rewards

            OUTPUTS:
            ------------
            - avg_rewards - (int) deque containing average rewards
            - best_avg_reward - (int) largest value in the avg_rewards deque
        """
        # initialize average rewards
        avg_rewards = deque(maxlen=num_episodes)
        # initialize best average reward
        best_avg_reward = -math.inf
        # initialize monitor for most recent rewards
        samp_rewards = deque(maxlen=window)
        # for each episode
        for i_episode in range(1, num_episodes+1):
            # begin the episode
            state = env.reset()
            # initialize the sampled reward
            samp_reward = 0
            while True:
                # agent selects an action
                action = agent.select_action(state, i_episode)
                # agent performs the selected action
                next_state, reward, done, _ = env.step(action)
                # agent performs internal updates based on sampled experience
                agent.step(state, action, reward, next_state, done)
                # update the sampled reward
                samp_reward += reward
                # update the state (s <- s') to next time step
                state = next_state
                if done:
                    # save final sampled reward
                    samp_rewards.append(samp_reward)
                    break
            if (i_episode >= 100):
                # get average reward from last 100 episodes
                avg_reward = np.mean(samp_rewards)
                # append to deque
                avg_rewards.append(avg_reward)
                # update best average reward
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
            # monitor progress
            print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
            # check if task is solved (according to OpenAI Gym)
            if best_avg_reward >= 9.7:
                print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
                break
            if i_episode == num_episodes: print('\n')
        
        return avg_rewards, best_avg_reward
    ```

## Implementation - main.py
- Open Python file ```main.py```
    ```
    from agent import Agent
    from monitor import interact
    import gym
    import numpy as np

    env = gym.make('Taxi-v2')
    agent = Agent()
    avg_rewards, best_avg_reward = interact(env, agent)
    ```


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```
### To Start taxi-v2 Training
- Open your terminal
- Navigate to ```main.py```
- Type in terminal
    ```
    python main.py
    ```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)