"""Pareto Q-Learning."""
import numbers
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import wandb
###new####

import gym

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value

def action_func_Mlb(actions):
        env_actions = []

        CIO = [-2, -1, 0, 1, 2]
       # CIO = [-3,-2, -1, 0, 1, 3]

        # Ttt_len = len(Ttt)
        # print("Ttt_len: ",Ttt_len)
        CIO_len = len(CIO)
        # print("Hom_len:  ",Hom_len)

        
            # print("i: ",i)
        env_action = actions%(CIO_len)
            # print(env_action)
        env_action = CIO[env_action]
        


            # print("Action index: ",i)
            # print("TTT: ",Ttt_action, " HOM: ", Hom_action)
            
        
            
        return env_action
    
    
    
# def action_func_Mlb(actions):
#     env_actions = []

#     CIO = [-2, -1, 0, 1, 2,3,4,-3,-4]
    

#     # Ttt_len = len(Ttt)
#     # print("Ttt_len: ",Ttt_len)
#     CIO_len = len(CIO)
#     # print("Hom_len:  ",Hom_len)

#     for i in range (3):
#         # print("i: ",i)
#         env_action = divmod(actions, CIO_len)
#         print(env_action)
#         CIO_action = CIO[env_action[1]]
#         actions = int(actions/9)

#         # print("Action index: ",i)
#         # print("TTT: ",Ttt_action, " HOM: ", Hom_action)
        
#         env_actions.append(CIO_action)
        
#     return env_actions



class PQL(MOAgent):
    """Pareto Q-learning.

    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """
    
    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Pareto Q-Learning",
        wandb_entity: Optional[str] = None,
        log: bool = True,
    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            log: Whether to log or not.
        """
        super().__init__(env, seed=seed)
        #super().__init__(seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.ref_point = ref_point
        #### new ##############################################################################################################################################################################################################################################################################################################################################
        print ("action_space_type", type(self.env.action_space))
        #########################################################################################################################################################################################################################################
        if type(self.env.action_space) == gym.spaces.Discrete:
            self.num_actions = self.env.action_space.n
        elif type(self.env.action_space) == gym.spaces.MultiDiscrete:
            self.num_actions = np.prod(self.env.action_space.nvec)
        ################ new ############################################################################################################################################################################################################################
        elif isinstance(self.env.action_space, gym.spaces.box.Box):
             self.num_actions = 7
            #  self.num_actions = 9*9*9
        ##################################################################################################################################################################################################################################################################     
        else :     
            raise Exception("PQL only supports (multi)discrete action spaces.")
        
        
        print ("observation_space_type", type(self.env.observation_space))
        if type(self.env.observation_space) == gym.spaces.Discrete:
            self.env_shape = (self.env.observation_space.n,)
        elif type(self.env.observation_space) == gym.spaces.MultiDiscrete:
            self.env_shape = self.env.observation_space.nvec
        ################### new ###########################################################################################################################################################################################################################################   
        elif isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            self.env_shape = (len(self.env.observation_space),)
        ###################################################################################################################################################################################################################################################################
        elif (
            type(self.env.observation_space) == gym.spaces.Box
            and self.env.observation_space.is_bounded(manner="both")
            and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        #self.num_states = np.prod(self.env_shape)
        self.num_states = 72
        #self.num_objectives = self.env.reward_space.shape[0]
        ############# new #############################################################################################################################################################################################################################
        self.num_objectives = 2
        ###############################################################################################################################################################################################################################################
        self.counts = np.zeros((self.num_states, self.num_actions))
       # self.non_dominated = [
       #      [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
       #  ]
        
        self.non_dominated =[
               [{tuple(np.full(self.num_objectives, -2))} for _ in range(self.num_actions)] #-2
                for _ in range(self.num_states)
            ]
        
        
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name, entity=wandb_entity)

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            #"env_id": self.env.unwrapped.spec.id,
            ##################### new ######################## 
            "env_id": 1109,
            ##################################################
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, state: int):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
  
        
       # print("state", state)
       # print("action",action)
        nd_array = np.array(list(self.non_dominated[state][action]))
        
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated
    

    def action_select(
        self,
        eval_env: gym.Env,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        action_eval: Optional[str] = "hypervolume",
        agentNumb: Optional[int] = 3
    ):
        """Learn the Pareto front.

        Args:
            total_timesteps (int, optional): The number of episodes to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
        """
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
        if ref_point is None:
            ref_point = self.ref_point
          #  print("hi")
        if self.log:
            self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})
           # print("hello")
            

        state = eval_env['enbMLBindicator'][agentNumb]
        state_int = [int(state)]
       # print("State",state_int)

        state_index = np.ravel_multi_index(state_int, (36))
       # print(state)
   

        action = self.select_action(state, score_func)
        # print("action",agentNumb," : " ,action )
        action1 = action_func_Mlb(action)

        return action1
    
    def train(
        self,
        obs,
        next_obs,
        action,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        action_eval: Optional[str] = "hypervolume",
        agentNumb: Optional[int] = 3
    ):
        
        obs1 = obs ['enbMLBindicator'][agentNumb]
        obs2 = obs ['enbneigborMLBindicator'][agentNumb]
        obs3 = obs ['RSRPIndicator'][agentNumb]
        obs4 = obs ['CQIcompareidicator'][agentNumb]
        print("-------------agent Num : ",agentNumb)
        print("enbMLBindicator : ",obs1)
        print("enbneigborMLBindicator : ",obs2)
        print("RSRPIndicator : ",obs3)
        print("CQIcompareidicator : ",obs4)
        print("--------------------------" )
        obs1_int = int(obs1)
        obs2_int = int(obs2)
        obs3_int = int(obs3)
        obs4_int = int(obs4)
        multi_indices = [obs1_int, obs2_int, obs3_int, obs4_int]
        dims = (9, 2, 2, 2)
        obs = np.ravel_multi_index(multi_indices, dims)
        print("obs" , obs)
        
        reward = []
        reward1 = next_obs['MLBreward'][agentNumb]
        reward2_1 = next_obs['MROreward'][0]
        reward2_2 = next_obs['MROreward'][1]
        reward2_3 = next_obs['MROreward'][2]
        reward2_4 = next_obs['MROreward'][3]
        reward2_5 = next_obs['MROreward'][4]
        reward2_6 = next_obs['MROreward'][5]
        reward2_7 = next_obs['MROreward'][6]
        reward2_8 = next_obs['MROreward'][7]
        reward2_9 = next_obs['MROreward'][8]
        reward2_10 = next_obs['MROreward'][9]
        reward2_11 = next_obs['MROreward'][10]
        reward2_12 = next_obs['MROreward'][11]
        reward2_13 = next_obs['MROreward'][12]
        
        if(agentNumb == 0 ):
            reward2 = reward2_1 + reward2_6 + reward2_7 
        elif(agentNumb == 1 ) :
            reward2 = reward2_2 + reward2_10 + reward2_11 
        elif(agentNumb == 2 ) :
            reward2 = reward2_3 + reward2_6 + reward2_7 + reward2_8 + reward2_9  + reward2_10 + reward2_11 + reward2_12 +reward2_13     
        elif(agentNumb == 3 ) :
            reward2 = reward2_4 + reward2_12 + reward2_13 
        elif(agentNumb == 4 ) :
            reward2 = reward2_5 + reward2_8 + reward2_9 
        elif(agentNumb == 5 ) :
            reward2 = reward2_6 + reward2_7 + reward2_3 + reward2_10 + reward2_1
        elif(agentNumb == 6 ) :
            reward2 = reward2_6 + reward2_7 + reward2_3 + reward2_1+ reward2_12
        elif(agentNumb == 7 ) :
            reward2 = reward2_8 + reward2_9 + reward2_3 + reward2_11 + reward2_5
        elif(agentNumb == 8 ) :
            reward2 = reward2_9 + reward2_8 + reward2_3 + reward2_13 + reward2_5
        elif(agentNumb == 9 ) :
            reward2 = reward2_10 + reward2_11 + reward2_2 + reward2_6 + reward2_3
        elif(agentNumb == 10 ) :
            reward2 = reward2_11 + reward2_10 + reward2_2 + reward2_3 + reward2_8 
        elif(agentNumb == 11 ) :
            reward2 = reward2_12 + reward2_13 + reward2_4 + reward2_3 + reward2_7
        elif(agentNumb == 12 ) :
            reward2 = reward2_13 + reward2_12 + reward2_4 + reward2_3 + reward2_9
                              
        
        reward.append(reward1)
        reward.append(reward2)
        print("reward ",agentNumb +1," : " ,reward)
        action = action[agentNumb]
        
        next_obs1 = next_obs ['enbMLBindicator'][agentNumb]
        next_obs2 = next_obs ['enbneigborMLBindicator'][agentNumb]
        next_obs3 = next_obs ['RSRPIndicator'][agentNumb]
        next_obs4 = next_obs ['CQIcompareidicator'][agentNumb]
        next_obs1_int = int(next_obs1)
        next_obs2_int = int(next_obs2)
        next_obs3_int = int(next_obs3)
        next_obs4_int = int(next_obs4)
        next_multi_indices = [next_obs1_int, next_obs2_int, next_obs3_int, next_obs4_int]
        dims = (9, 2, 2, 2)
        next_obs = np.ravel_multi_index(next_multi_indices, dims)
        # print("MLBstate : " ,next_state_int[1])
        # next_state = np.ravel_multi_index(next_state_int, (10,  12))

        
        ################################################################
        self.counts[obs, action] += 1
      #  print("obs : ", obs)
       # print("aciton : ",action)
       # print("next_obs : ",next_obs)
        self.non_dominated[obs][action] = self.calc_non_dominated(next_obs)
        #print("train : ", agentNumb +1)
       # print("self.avg_reward[state, action]", self.avg_reward[obs, action])
        #print("non_dominate : ", self.non_dominated[obs][action])
    #     for obs in range(len(self.non_dominated)):#
    #         for action in range(len(self.non_dominated[obs])):
    #             value = self.non_dominated[obs][action]
    #    #        print(f"Value at state {obs} and action {action}: {value}")
        self.avg_reward[obs, action] += (reward - self.avg_reward[obs, action]) / self.counts[obs, action]
        # print("average reward: " ,self.avg_reward)

        obs = next_obs
        
        
    

        return self.get_local_pcs(state=obs)

    def _eval_all_policies(self, env: gym.Env, agentNumb) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env, agentNumb))
            

        return pf
    

   # def track_policy_action_selction1(self, vec, env: gym.Env, next_obs, agentNumb, tol=1e-3):
    def track_policy_action_selction1(self, vec, next_obs, agentNumb, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
   
        target = np.array(vec)
        #state = next_obs['enbMLBstate'][agentNumb]
        #state = [int(state)]
        #######################################
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0
        total_step = 0

    
        #state = np.ravel_multi_index(state, (36))
        
        obs1 = next_obs ['enbMLBindicator'][agentNumb]
        obs2 = next_obs ['enbneigborMLBindicator'][agentNumb]
        obs3 = next_obs['RSRPIndicator'][agentNumb]
        obs4 = next_obs ['CQIcompareidicator'][agentNumb]
        print("-------------agent Num : ",agentNumb)
        print("enbMLBindicator : ",obs1)
        print("enbneigborMLBindicator : ",obs2)
        print("RSRPIndicator : ",obs3)
        print("CQIcompareidicator : ",obs4)
        print("--------------------------" )
        obs1_int = int(obs1)
        obs2_int = int(obs2)
        obs3_int = int(obs3)
        obs4_int = int(obs4)
        multi_indices = [obs1_int, obs2_int, obs3_int, obs4_int]
        dims = (9, 2, 2, 2)
        state = np.ravel_multi_index(multi_indices, dims)
        
        
        closest_dist = np.inf
        closest_action = 0
        found_action = False
        new_target = target
     #   print("self.num_actions : " ,self.num_actions)
        

        closest_dist = np.inf

        for action in range(self.num_actions):
            im_rew = self.avg_reward[state, action]
            non_dominated_set = self.non_dominated[state][action]

            if not non_dominated_set:
                # If non_dominated_set is empty, consider this action
                dist = 0  # You can set it to 0 or any other value depending on your criteria
            else:
                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))

            if dist < closest_dist:
                closest_dist = dist
                closest_action = action
                new_target = q

                if dist < tol:
                    found_action = True
                    closest_action = action_func_Mlb(closest_action)
                    print("Selected action for Agent", agentNumb + 1, " : ", closest_action)
                    print("Selected new_target for Agent", agentNumb + 1, " : ", new_target)
                    result = []
                    result.append(closest_action)
                    result.append(new_target)
                    break

        if not found_action:
            closest_action = action_func_Mlb(closest_action)
            print("Selected action for Agent", agentNumb + 1, " : ", closest_action)
            print("Selected new_target for Agent", agentNumb + 1, " : ", new_target)
            result = []
            result.append(closest_action)
            result.append(new_target)
            

        return result
            
          

        return result
    
    
   # def track_policy_action_selction2(self, vec, env: gym.Env, next_obs, agentNumb, target, tol=1e-3):
    def track_policy_action_selction2(self, next_obs, agentNumb, target, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
        
        print("target : ",target)

        state = next_obs['enbMLBstate'][agentNumb]
        state = [int(state)]
        #######################################

    
        state = np.ravel_multi_index(state, (36))
        closest_dist = np.inf
        closest_action = 0
        found_action = False
        new_target = target
        #print("self.num_actions : " ,self.num_actions)
        

        
        closest_dist = np.inf

        for action in range(self.num_actions):
            im_rew = self.avg_reward[state, action]
            non_dominated_set = self.non_dominated[state][action]

            if not non_dominated_set:
                # If non_dominated_set is empty, consider this action
                dist = 0  # You can set it to 0 or any other value depending on your criteria
            else:
                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))

            if dist < closest_dist:
                closest_dist = dist
                closest_action = action
                new_target = q

                if dist < tol:
                    found_action = True
                    closest_action = action_func_Mlb(closest_action)
                    print("Selected action for Agent", agentNumb + 1, " : ", closest_action)
                    print("Selected new_target for Agent", agentNumb + 1, " : ", new_target)
                    result = []
                    result.append(closest_action)
                    result.append(new_target)
                    break

        if not found_action:
            closest_action = action_func_Mlb(closest_action)
            print("Selected action for Agent", agentNumb + 1, " : ", closest_action)
            print("Selected new_target for Agent", agentNumb + 1, " : ", new_target)
            result = []
            result.append(closest_action)
            result.append(new_target)
            

        return result
    



    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.
        print("XXXXXXXXXXXXXX")
        
        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)
