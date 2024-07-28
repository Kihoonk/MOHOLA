#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numbers
from typing import Callable, List, Optional
import mo_gymnasium as mo_gym
import argparse
from ns3gym import ns3env
# from action_func import*
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import wandb
import csv

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.common.utils import linearly_decaying_value




startSim = False
iterationNum = 200
port = 1210
simTime = 15 # seconds
stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--stepTime": stepTime,
           "--testArg": 123}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space          
# print("Observation space: ", ob_space,  ob_space.dtype)
# print("Action space: ", ac_space, ac_space.dtype)
# print(" ********ovservation space********* ",env.observation_space.shape)
# print("**********************************************")


stepIdx = 0
currIt = 0
RlfCount = []
PingpongCount = []
ref_point = np.array([0, 0])





agent1 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=5,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent2 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=4,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent3 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=6,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent4 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=2,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent5 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=1,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent6 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=1,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent7 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=7,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent8 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=8,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent9 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=9,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent10 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=10,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent11 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=11,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent12 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=12,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

agent13 = PQL(
    env,
    ref_point,
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=50000,
    final_epsilon=0.2,
    seed=13,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    log=True,
)

try:
    while True:
        # print("hi")
        print("Start iteration: ", currIt)
        obs = env.reset()
        print("Step: ", stepIdx)
        # print("---obs: ", obs)
        
        if(stepIdx<2):
            numOfenb = 13

        while True:
            if(stepIdx == 0) :
                prev_action=[0,0,0,0,0,0,0,0,0,0,0,0,0]
                obs = env.reset()
            print("---------------------------------------------- Step ", stepIdx , "----------------------------------------------" )
            stepIdx += 1
            
            env1 = obs
            env2 = obs
            env3 = obs
            env4 = obs
            env5 = obs
            env6 = obs
            env7 = obs
            env8 = obs
            env9 = obs
            env10 = obs
            env11 = obs
            env12 = obs
            env13 = obs

            state_velocity = np.reshape(obs['AverageVelocity'], [13,1])
            state_edge = np.reshape(obs['FarUes'], [13,1])
            action_mro=[]
            #print("state_velocity :",  state_velocity)
            #print("state_edge :",  state_edge)
            for i in range(13):
                if (state_velocity[i] < 20) :
                    if (state_edge[i]<0.3) :
                        action_mro.append(3)
                        action_mro.append(256)
                    elif(state_edge[i]>=0.3) :
                        action_mro.append(6)
                        action_mro.append(512)
                elif(state_velocity[i]<35) :
                    if (state_edge[i]<0.3) :
                        action_mro.append(2)
                        action_mro.append(128)
                    elif(state_edge[i]>=0.3) :
                        action_mro.append(4)
                        action_mro.append(320)
                else :
                    if (state_edge[i]<0.3) :
                        action_mro.append(1)
                        action_mro.append(64)
                    elif(state_edge[i]>=0.3) :
                        action_mro.append(2)
                        action_mro.append(128)
                            

 ############################################################# Multi Agent Action ######################################################################
            if((stepIdx%3)==1):
                
                action1 = agent1.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env1,
                    agentNumb = 0
                )
                
                action2 = agent2.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env2,
                    agentNumb = 1
                )
                
                action3 = agent3.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env3,
                    agentNumb = 2
                )
                
                action4 = agent4.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env4,
                    agentNumb = 3
                )
                
                action5 = agent5.action_select(
                action_eval="hypervolume",
                known_pareto_front = None,
                ref_point=ref_point,
                eval_env=env5,
                agentNumb = 4
                )  
                
                action6 = prev_action[5]
                action7 = prev_action[6]
                action8 = prev_action[7]
                action9 = prev_action[8]
                action10 = prev_action[9]
                action11 = prev_action[10]
                action12 = prev_action[11]
                action13 = prev_action[12]
            
            if((stepIdx%3)==2):
                
                action6 = agent6.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env6,
                    agentNumb = 5
                )
                
                
                action9 = agent9.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env9,
                    agentNumb = 8
                )   
                
                action11 = agent11.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env11,
                    agentNumb = 10
                )     
                
                action12 = agent12.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env12,
                    agentNumb = 11
                )    
                
                action1 = prev_action[0]
                action2 = prev_action[1]
                action3 = prev_action[2]
                action4 = prev_action[3]
                action5 = prev_action[4]
                action7 = prev_action[6]
                action8 = prev_action[7]
                action10 = prev_action[9]
                action13 = prev_action[12]
                
            if((stepIdx%3)==0):
                
                action7 = agent7.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env7,
                    agentNumb = 6
                )
                
                
                action8 = agent8.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env8,
                    agentNumb = 7
                )   
                
                action10 = agent10.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env10,
                    agentNumb = 9
                )     
                
                action13 = agent13.action_select(
                    action_eval="hypervolume",
                    known_pareto_front = None,
                    ref_point=ref_point,
                    eval_env=env13,
                    agentNumb = 12
                )    
                
                action1 = prev_action[0]
                action2 = prev_action[1]
                action3 = prev_action[2]
                action4 = prev_action[3]
                action5 = prev_action[4]
                action6 = prev_action[5]
                action9 = prev_action[8]
                action11 = prev_action[10]
                action12 = prev_action[11]
                
            
            action_mlb =[]


            
 
            action_mlb.append(action1)
            action_mlb.append(action2)
            action_mlb.append(action3)
            action_mlb.append(action4)
            action_mlb.append(action5)
            action_mlb.append(action6)
            action_mlb.append(action7)
            action_mlb.append(action8)
            action_mlb.append(action9)
            action_mlb.append(action10)
            action_mlb.append(action11)
            action_mlb.append(action12)
            action_mlb.append(action13)
            
            action =[]
            
            for m in action_mlb:
                action.append(m)
            for n in action_mro:
                action.append(n)
     

           
           
           
            print("action_mro: ", action_mro)
            print("action_mlb: ", action_mlb)

            # MRO_alg
            
            next_obs, reward, done, info = env.step(action)
            
            # MRO_nonalg
            # nonalg_action = []
            # for i in range(numOfenb) :
            #     nonalg_action.append(0)
            # obs, reward, done, info = env.step(nonalg_action)
            if (stepIdx==28) :
            #if(stepIdx == 7) :  
                stepIdx=0
            
 ############################################################## learning parts ########################################################################
            print("step test : ",stepIdx)
            if(stepIdx>1):
                
                if((stepIdx%3)==1):
                    print("Cluster 1 learning")
                    
                    agent5.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 4,
                    )
                    
                    agent1.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 0,
                    )
                    
                    agent2.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 1,
                    )
                    
                    agent3.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 2,
                    )
                    
                    agent4.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 3,
                    )
                    
                    
                if((stepIdx%3)==2):
                    print("Cluster 2 learning")
                    agent6.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 5,
                    )
                

                    agent9.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 8,
                    )
                    
                    agent11.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 10,
                    )
                    
                    agent12.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 11,
                    )
                    
                if((stepIdx%3)==0):  
                    print("Cluster 3 learning") 
                    
                    agent7.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 6,
                    )
                

                    agent8.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 7,
                    )
                    
                    agent10.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 9,
                    )
                    
                    agent13.train(
                        obs,
                        next_obs,
                        action,
                        action_eval="hypervolume",
                        known_pareto_front = None,
                        ref_point=ref_point,
                        agentNumb = 12,
                    )

                
                
            # print("obs : ",obs)
            # print("done : ",done)
            # print("---obs, reward, done, info: ", next_obs, reward, done, info)
            # print("done : ",done)
            # print("stepIdx : ",stepIdx)

            obs = next_obs
            prev_action[0] = action1
            prev_action[1] = action2
            prev_action[2] = action3
            prev_action[3] = action4
            prev_action[4] = action5
            prev_action[5] = action6
            prev_action[6] = action7
            prev_action[7] = action8
            prev_action[8] = action9
            prev_action[9] = action10
            prev_action[10] = action11
            prev_action[11] = action12
            prev_action[12] = action13
            

            if (currIt==0 and stepIdx==27):
                f = open('MroCase_pareto_paper.txt','w',encoding='utf-8')
                f.write(str(Results[0])+"\n")
                f.write(str(Results[1])+"\n"+"\n")
                f.close()
            elif (currIt!=0 and stepIdx==27):
                f = open('MroCase_pareto_paper.txt','a',encoding='utf-8')
                f.write(str(Results[0])+"\n")
                f.write(str(Results[1])+"\n"+"\n")
                f.close()
            
         
            if(stepIdx == 27) :
            #if(stepIdx == 6) :    
                done = True
                currIt += 1

            else:
                done = False
                
            if done:
                print("decaying")

                    
                agent1.epsilon = linearly_decaying_value(
                agent1.initial_epsilon,
                agent1.epsilon_decay_steps,
                agent1.global_step,
                0,
                agent1.final_epsilon,
                )
                
                agent2.epsilon = linearly_decaying_value(
                agent2.initial_epsilon,
                agent2.epsilon_decay_steps,
                agent2.global_step,
                0,
                agent2.final_epsilon,
                )
                
                agent3.epsilon = linearly_decaying_value(
                agent3.initial_epsilon,
                agent3.epsilon_decay_steps,
                agent3.global_step,
                0,
                agent3.final_epsilon,
                )
                
                agent4.epsilon = linearly_decaying_value(
                agent4.initial_epsilon,
                agent4.epsilon_decay_steps,
                agent4.global_step,
                0,
                agent4.final_epsilon,
                )
                
                agent5.epsilon = linearly_decaying_value(
                agent5.initial_epsilon,
                agent5.epsilon_decay_steps,
                agent5.global_step,
                0,
                agent5.final_epsilon,
                )
                
                agent6.epsilon = linearly_decaying_value(
                agent6.initial_epsilon,
                agent6.epsilon_decay_steps,
                agent6.global_step,
                0,
                agent6.final_epsilon,
                )
                
                agent7.epsilon = linearly_decaying_value(
                agent7.initial_epsilon,
                agent7.epsilon_decay_steps,
                agent7.global_step,
                0,
                agent7.final_epsilon,
                )
                
                agent8.epsilon = linearly_decaying_value(
                agent8.initial_epsilon,
                agent8.epsilon_decay_steps,
                agent8.global_step,
                0,
                agent8.final_epsilon,
                )
                
                agent9.epsilon = linearly_decaying_value(
                agent9.initial_epsilon,
                agent9.epsilon_decay_steps,
                agent9.global_step,
                0,
                agent9.final_epsilon,
                )
                
                agent10.epsilon = linearly_decaying_value(
                agent10.initial_epsilon,
                agent10.epsilon_decay_steps,
                agent10.global_step,
                0,
                agent10.final_epsilon,
                )
                
                agent11.epsilon = linearly_decaying_value(
                agent11.initial_epsilon,
                agent11.epsilon_decay_steps,
                agent11.global_step,
                0,
                agent11.final_epsilon,
                )
                
                agent12.epsilon = linearly_decaying_value(
                agent12.initial_epsilon,
                agent12.epsilon_decay_steps,
                agent12.global_step,
                0,
                agent12.final_epsilon,
                )
                
                agent13.epsilon = linearly_decaying_value(
                agent13.initial_epsilon,
                agent13.epsilon_decay_steps,
                agent13.global_step,
                0,
                agent13.final_epsilon,
                )
                
                done_policy = False
                
  ###################################################################### track policy #######################################################################################################################         
                print("iternum : ",currIt)              
            if ((stepIdx==0)&(currIt==100)):
                print("------------------------------------------------------------------ Tracking  policy --------------------------------------------------------------------------")
                prev_action_tracking=[0]*13
                while not done_policy:
                    print("EP : ", currIt)
                    
                    if (stepIdx == 0):
                        env.reset() 
                        total_rew1 = np.zeros(2)
                        total_rew2 = np.zeros(2)
                        total_rew3 = np.zeros(2)
                        total_rew4 = np.zeros(2)
                        total_rew5 = np.zeros(2)
                        total_rew6 = np.zeros(2)
                        total_rew7 = np.zeros(2)
                        total_rew8 = np.zeros(2)
                        total_rew9 = np.zeros(2)
                        total_rew10 = np.zeros(2)
                        total_rew11 = np.zeros(2) 
                        total_rew12 = np.zeros(2)
                        total_rew13 = np.zeros(2)

                        current_gamma = 1.0 
                                                
                    elif (stepIdx == 1):

                        action1 = agent1.track_policy_action_selction1(max(agent1.get_local_pcs(state=int(state1)), key=lambda x: x[1]), next_obs, 0)
                        action2 = agent2.track_policy_action_selction1(max(agent2.get_local_pcs(state=int(state2)), key=lambda x: x[1]), next_obs, 1)
                        action3 = agent3.track_policy_action_selction1(max(agent3.get_local_pcs(state=int(state3)), key=lambda x: x[1]), next_obs, 2)
                        action4 = agent4.track_policy_action_selction1(max(agent4.get_local_pcs(state=int(state4)), key=lambda x: x[1]), next_obs, 3)
                        action5 = agent5.track_policy_action_selction1(max(agent5.get_local_pcs(state=int(state5)), key=lambda x: x[1]), next_obs, 4)
                        action6 = agent6.track_policy_action_selction1(max(agent6.get_local_pcs(state=int(state6)), key=lambda x: x[1]), next_obs, 5)
                        action7 = agent7.track_policy_action_selction1(max(agent7.get_local_pcs(state=int(state7)), key=lambda x: x[1]), next_obs, 6)
                        action8 = agent8.track_policy_action_selction1(max(agent8.get_local_pcs(state=int(state8)), key=lambda x: x[1]), next_obs, 7)
                        action9 = agent9.track_policy_action_selction1(max(agent9.get_local_pcs(state=int(state9)), key=lambda x: x[1]), next_obs, 8)
                        action10 = agent10.track_policy_action_selction1(max(agent10.get_local_pcs(state=int(state10)), key=lambda x: x[1]), next_obs, 9)
                        action11 = agent11.track_policy_action_selction1(max(agent11.get_local_pcs(state=int(state11)), key=lambda x: x[1]), next_obs, 10)
                        action12 = agent12.track_policy_action_selction1(max(agent12.get_local_pcs(state=int(state12)), key=lambda x: x[1]), next_obs, 11)
                        action13 = agent13.track_policy_action_selction1(max(agent13.get_local_pcs(state=int(state13)), key=lambda x: x[1]), next_obs, 12)
                        
                    else :
                        if((stepIdx%3)==1):
                            action1 = agent1.track_policy_action_selction1(max(agent1.get_local_pcs(state=int(state1)), key=lambda x: x[1]), next_obs, 0)
                            action2 = agent2.track_policy_action_selction1(max(agent2.get_local_pcs(state=int(state2)), key=lambda x: x[1]), next_obs, 1)
                            action3 = agent3.track_policy_action_selction1(max(agent3.get_local_pcs(state=int(state3)), key=lambda x: x[1]), next_obs, 2)
                            action4 = agent4.track_policy_action_selction1(max(agent4.get_local_pcs(state=int(state4)), key=lambda x: x[1]), next_obs, 3)
                            action5 = agent5.track_policy_action_selction1(max(agent5.get_local_pcs(state=int(state5)), key=lambda x: x[1]), next_obs, 4)       
                            action6=prev_action_tracking[5]
                            action7=prev_action_tracking[6]
                            action8=prev_action_tracking[7]
                            action9=prev_action_tracking[8]
                            action10=prev_action_tracking[9]
                            action11=prev_action_tracking[10]
                            action12=prev_action_tracking[11]
                            action13=prev_action_tracking[12]
                            
                            
                        if((stepIdx%3)==2):    
                            action6 = agent6.track_policy_action_selction1(max(agent6.get_local_pcs(state=int(state6)), key=lambda x: x[1]), next_obs, 5)
                            action9 = agent9.track_policy_action_selction1(max(agent9.get_local_pcs(state=int(state9)), key=lambda x: x[1]), next_obs, 8)
                            action11 = agent10.track_policy_action_selction1(max(agent10.get_local_pcs(state=int(state10)), key=lambda x: x[1]), next_obs, 9)
                            action12 = agent11.track_policy_action_selction1(max(agent11.get_local_pcs(state=int(state11)), key=lambda x: x[1]), next_obs, 10)
                            print("best_pcs_2 for mlb  : ", max(agent2.get_local_pcs(state = int(state2)), key=lambda x: x[0]))
                            print("best_pcs_4 for mlb  : ", max(agent4.get_local_pcs(state = int(state4)), key=lambda x: x[0]))
                        
                            action1=prev_action_tracking[0]
                            action2=prev_action_tracking[1]
                            action3=prev_action_tracking[2]
                            action4=prev_action_tracking[3]
                            action5=prev_action_tracking[4]
                            action7=prev_action_tracking[6]
                            action8=prev_action_tracking[7]
                            action10=prev_action_tracking[9]
                            action13=prev_action_tracking[12]
                            
                        if((stepIdx%3)==0): 
                            
                            action7 = agent7.track_policy_action_selction1(max(agent7.get_local_pcs(state=int(state7)), key=lambda x: x[1]), next_obs, 6)
                            action8 = agent8.track_policy_action_selction1(max(agent8.get_local_pcs(state=int(state8)), key=lambda x: x[1]), next_obs, 7)
                            action10 = agent10.track_policy_action_selction1(max(agent10.get_local_pcs(state=int(state10)), key=lambda x: x[1]), next_obs, 9)
                            action13 = agent13.track_policy_action_selction1(max(agent13.get_local_pcs(state=int(state13)), key=lambda x: x[1]), next_obs, 12)

                            print("best_pcs_3 for mlb  : ", max(agent3.get_local_pcs(state = int(state3)), key=lambda x: x[0]))
                            
                            action1=prev_action_tracking[0]
                            action2=prev_action_tracking[1]
                            action3=prev_action_tracking[2]
                            action4=prev_action_tracking[3]
                            action5=prev_action_tracking[4]
                            action6=prev_action_tracking[5]
                            action9=prev_action_tracking[8]
                            action11=prev_action_tracking[10]
                            action12=prev_action_tracking[11]
                            
                            
                    if (stepIdx ==0) :
                        
                        actions1 = 0
                        actions2 = 0
                        actions3 = 0
                        actions4 = 0
                        actions5 = 0
                        actions6 = 0
                        actions7 = 0
                        actions8 = 0
                        actions9 = 0
                        actions10= 0
                        actions11 = 0
                        actions12 = 0
                        actions13 = 0
                        actions=[0,0,0,0,0,0,0,0,0,0,0,0,0,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3,320,3]
                        
                    else :

                        if (isinstance(action1, int)==0):
                            actions1= action1[0]
                        else :
                            actions1 = action1
              
                        if (isinstance(action2, int)==0):    
                            actions2= action2[0]
                        else :
                            actions2 = action2
                
                        if (isinstance(action3, int)==0):    
                            actions3= action3[0]
                        else :
                            actions3 = action3
                            
                        if (isinstance(action4, int)==0):   
                            actions4= action4[0]
                        else :
                            actions4 = action4

                        if (isinstance(action5, int)==0):   
                            actions5= action5[0]  
                        else :
                            actions5 = action5

                        if (isinstance(action6, int)==0):   
                            actions6= action6[0]  
                        else :
                            actions6 = action6
                            
                        if (isinstance(action7, int)==0):   
                            actions7= action7[0] 
                        else :
                            actions7 = action7
                        
                        if (isinstance(action8, int)==0):   
                            actions8= action8[0]  
                        else :
                            actions8 = action8
                            
                        if (isinstance(action9, int)==0):   
                            actions9= action9[0]  
                        else :
                            actions9 = action9
                            
                        if (isinstance(action10, int)==0):   
                            actions10= action10[0]  
                        else :
                            actions10 = action10
                            
                        if (isinstance(action11, int)==0):   
                            actions11= action11[0]  
                        else :
                            actions11 = action11
                            
                        if (isinstance(action12, int)==0):   
                            actions12= action12[0]  
                        else :
                            actions12 = action12
                            
                        if (isinstance(action13, int)==0):   
                            actions13= action13[0]  
                        else :
                            actions13 = action13
                            
                        
                            
                        actions=[]                     
                        actions_mlb=[]
                        actions_mlb.append(actions1)
                        actions_mlb.append(actions2)
                        actions_mlb.append(actions3)
                        actions_mlb.append(actions4)
                        actions_mlb.append(actions5)
                        actions_mlb.append(actions6)
                        actions_mlb.append(actions7)
                        actions_mlb.append(actions8)
                        actions_mlb.append(actions9)
                        actions_mlb.append(actions10)
                        actions_mlb.append(actions11)
                        actions_mlb.append(actions12)
                        actions_mlb.append(actions13)
                        
                        actions_mro =[]        
                        state_velocity =[]
                        
                        state_velocity.append(state_velocity1)
                        state_velocity.append(state_velocity2)
                        state_velocity.append(state_velocity3)
                        state_velocity.append(state_velocity4)
                        state_velocity.append(state_velocity5)
                        state_velocity.append(state_velocity6)
                        state_velocity.append(state_velocity7)
                        state_velocity.append(state_velocity8)
                        state_velocity.append(state_velocity9)
                        state_velocity.append(state_velocity10)
                        state_velocity.append(state_velocity11)
                        state_velocity.append(state_velocity12)
                        state_velocity.append(state_velocity13)
                        
                        state_edge =[]
                        state_edge.append(state_edge1)
                        state_edge.append(state_edge2)
                        state_edge.append(state_edge3)
                        state_edge.append(state_edge4)
                        state_edge.append(state_edge5)
                        state_edge.append(state_edge6)
                        state_edge.append(state_edge7)
                        state_edge.append(state_edge8)
                        state_edge.append(state_edge9)
                        state_edge.append(state_edge10)
                        state_edge.append(state_edge11)
                        state_edge.append(state_edge12)
                        state_edge.append(state_edge13)

                        
                        for i in range(13):
                            if (state_edge[i] < 20) :
                                if (state_edge[i] <30 ): 
                                    actions_mro.append (3)
                                    actions_mro.append (480)
                                elif (state_edge[i]>=30) :
                                    actions_mro.append (6)
                                    actions_mro.append (512)
                            elif(state_edge[i]<35) :
                                if (state_edge[i] <30 ): 
                                    actions_mro.append (2)
                                    actions_mro.append (256)
                                elif (state_edge[i]>=30):
                                    actions_mro.append (4)
                                    actions_mro.append (320)
                            else :
                                if (state_edge[i] <30 ): 
                                    actions_mro.append (1)
                                    actions_mro.append (64)
                                elif (state_edge[i]>=30):
                                    actions_mro.append (2)
                                    actions_mro.append (128)
   

                    
                    print("action : ",actions)
                    next_obs, reward, done, info = env.step(actions)
                    
                    reward1_1 = next_obs['MLBreward'][0]
                    reward1_2 = next_obs['MLBreward'][1]
                    reward1_3 = next_obs['MLBreward'][2]
                    reward1_4 = next_obs['MLBreward'][3]
                    reward1_5 = next_obs['MLBreward'][4]
                    reward1_6 = next_obs['MLBreward'][5]
                    reward1_7 = next_obs['MLBreward'][6]
                    reward1_8 = next_obs['MLBreward'][7]
                    reward1_9 = next_obs['MLBreward'][8]
                    reward1_10 = next_obs['MLBreward'][9]
                    reward1_11 = next_obs['MLBreward'][10]
                    reward1_12 = next_obs['MLBreward'][11]
                    reward1_13 = next_obs['MLBreward'][12]
                    
                    
                    
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
                    
                    reward1 = np.array([reward1_1, reward2_1])
                    reward2 = np.array([reward1_2, reward2_2])
                    reward3 = np.array([reward1_3, reward2_3])
                    reward4 = np.array([reward1_4, reward2_4])
                    reward5 = np.array([reward1_5, reward2_5])
                    reward6 = np.array([reward1_6, reward2_6])
                    reward7 = np.array([reward1_7, reward2_7])
                    reward8 = np.array([reward1_8, reward2_8])
                    reward9 = np.array([reward1_9, reward2_9])
                    reward10 = np.array([reward1_10, reward2_10])
                    reward11 = np.array([reward1_11, reward2_11])
                    reward12 = np.array([reward1_12, reward2_12])
                    reward13 = np.array([reward1_13, reward2_13])

                    
                    if(stepIdx==1):
                        target1 = action1[1]
                        target2 = action2[1]
                        target3 = action3[1]
                        target4 = action4[1]
                        target5 = action5[1]
                        target6 = action6[1]
                        target7 = action7[1]
                        target8 = action8[1]
                        target9 = action9[1]
                        target10= action10[1]
                        target11 = action11[1]
                        target12 = action12[1]
                        target13 = action13[1]

                        
                    if((stepIdx%3)==1):
                        total_rew1 += current_gamma * reward1
                        total_rew2 += current_gamma * reward2
                        total_rew3 += current_gamma * reward3
                        total_rew4 += current_gamma * reward4
                        total_rew5 += current_gamma * reward5
                        
                        target1 = action1[1]
                        target2 = action2[1]
                        target3 = action3[1]
                        target4 = action4[1]
                        target5 = action5[1]
                        
                    if((stepIdx%3)==2):    
                        total_rew6 += current_gamma * reward6
                        total_rew9 += current_gamma * reward9
                        total_rew11 += current_gamma * reward11
                        total_rew12 += current_gamma * reward12
                        target6 = action6[1]
                        target9 = action9[1]
                        target11 = action11[1]
                        target12 = action12[1]
                        
                        
                    if((stepIdx%3)==0):    
                        total_rew7 += current_gamma * reward7
                        total_rew8 += current_gamma * reward8
                        total_rew10 += current_gamma * reward10
                        total_rew13 += current_gamma * reward13     
                        target7 = action7[1]
                        target8 = action8[1]
                        target10 = action13[1]
                        target13 = action13[1]

                    #env = next_obs
                    stepIdx+=1;
                    prev_action_tracking[0] = actions1
                    prev_action_tracking[1] = actions2
                    prev_action_tracking[2] = actions3
                    prev_action_tracking[3] = actions4
                    prev_action_tracking[4] = actions5
                    prev_action_tracking[5] = actions6
                    prev_action_tracking[6] = actions7
                    prev_action_tracking[7] = actions8
                    prev_action_tracking[8] = actions9
                    prev_action_tracking[9] = actions10
                    prev_action_tracking[10] = actions11
                    prev_action_tracking[11] = actions12
                    prev_action_tracking[12] = actions13
                    
                 
                    for i in range(13):
                        obs1 = next_obs['enbMLBindicator'][i]
                        obs2 = next_obs['enbneigborMLBindicator'][i]
                        obs3 = next_obs['RSRPIndicator'][i]
                        obs4 = next_obs['CQIcompareidicator'][i]
                        obs1_int = int(obs1)
                        obs2_int = int(obs2)
                        obs3_int = int(obs3)
                        obs4_int = int(obs4)
                        multi_indices = [obs1_int, obs2_int, obs3_int, obs4_int]
                        dims = (9, 2, 2, 2)
                        if i == 0:
                            state1 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 1:
                            state2 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 2:
                            state3 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 3:
                            state4 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 4:
                            state5 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 5:
                            state6 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 6:
                            state7 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 7:
                            state8 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 8:
                            state9 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 9:
                            state10 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 10:
                            state11 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 11:
                            state12 = np.ravel_multi_index(multi_indices, dims)
                        elif i == 12:
                            state13 = np.ravel_multi_index(multi_indices, dims)
                        
                                                                                                                                                            
                    # state1= next_obs['enbMLBstate'][0]
                    # state2= next_obs['enbMLBstate'][1]
                    # state3= next_obs['enbMLBstate'][2]
                    # state4= next_obs['enbMLBstate'][3]
                    # state5= next_obs['enbMLBstate'][4]
                    # state6= next_obs['enbMLBstate'][5]
                    # state7= next_obs['enbMLBstate'][6]
                    # state8= next_obs['enbMLBstate'][7]
                    # state9= next_obs['enbMLBstate'][8]
                    # state10= next_obs['enbMLBstate'][9]
                    # state11= next_obs['enbMLBstate'][10]
                    # state12= next_obs['enbMLBstate'][11]
                    # state13= next_obs['enbMLBstate'][12]
                    
                    state_velocity1 = int(next_obs['AverageVelocity'][0])
                    state_velocity2 = int(next_obs['AverageVelocity'][1])
                    state_velocity3 = int(next_obs['AverageVelocity'][2])
                    state_velocity4 = int(next_obs['AverageVelocity'][3])
                    state_velocity5 = int(next_obs['AverageVelocity'][4])
                    state_velocity6 = int(next_obs['AverageVelocity'][5])
                    state_velocity7 = int(next_obs['AverageVelocity'][6])
                    state_velocity8 = int(next_obs['AverageVelocity'][7])
                    state_velocity9 = int(next_obs['AverageVelocity'][8])
                    state_velocity10 = int(next_obs['AverageVelocity'][9])
                    state_velocity11 = int(next_obs['AverageVelocity'][10])
                    state_velocity12 = int(next_obs['AverageVelocity'][11])
                    state_velocity13 = int(next_obs['AverageVelocity'][12])
                    
                    state_edge1 = int(next_obs['FarUes'][0])
                    state_edge2 = int(next_obs['FarUes'][1])
                    state_edge3 = int(next_obs['FarUes'][2])
                    state_edge4 = int(next_obs['FarUes'][3])
                    state_edge5 = int(next_obs['FarUes'][4])
                    state_edge6 = int(next_obs['FarUes'][5])
                    state_edge7 = int(next_obs['FarUes'][6])
                    state_edge8 = int(next_obs['FarUes'][7])
                    state_edge9 = int(next_obs['FarUes'][8])
                    state_edge10 = int(next_obs['FarUes'][9])
                    state_edge11 = int(next_obs['FarUes'][10])
                    state_edge12 = int(next_obs['FarUes'][11])
                    state_edge13 = int(next_obs['FarUes'][12])
                    
                    state_load = next_obs['rbUtil']
                    load1 = state_load[0] + state_load[1] +state_load[2] + state_load[3] +state_load[4]
                    load2 = state_load[0] + state_load[0] + state_load[0] + state_load[0]

                        
                    
                    print(" --------------------------------------- step : ",stepIdx,"  ---------------------------------------")
                   # if(stepIdx == 27) :
                    if(stepIdx == 28) :
                        done_policy = True
                        stepIdx = 0
                        currIt+=1
                        break
                                
                    
            
            
            if currIt == iterationNum:
                
                break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")