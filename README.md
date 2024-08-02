
MOHOLA README
================================

## OS and software preparation:

We base our experiment environment on Ubuntu 20.04 LTS and highly recommend that you do the same. This streamlines the setup process and avoids unexpected issues cause by incompatible software versions etc. Please make sure that you have Python installed. Also make sure that you have root or sudo permission.

This branch contains the entire ns-3 network simulator (ns-3.33) with ns3-gym (opengym) module.

## Install ns-3 and dependencies

1. The first part of the preparation is to clone the repository:

```shell
git clone https://github.com/Kihoonk/MOHOLA.git
```

2. Next, install all dependencies required by ns-3.

```shell
apt-get install gcc g++ python python3-pip
```

3. Install ZMQ and Protocol Buffers libs:

```shell
sudo apt-get update
apt-get install libzmq5 libzmq3-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
```

4. Install Pytorch.

```shell
pip3 install torch
```

Following guideline of installation in https://pytorch.org

5. Building ns-3

```shell
chmod +x ./waf
./waf configure
./waf build
```

6. Install ns3-gym

```shell
pip3 install --user ./src/opengym/model/ns3gym
```
7. Install MO-Gymnasium
     pip install mo-gymnasium
((https://github.com/LucasAlegre/morl-baselines))
@inproceedings{felten_toolkit_2023,
	author = {Felten, Florian and Alegre, Lucas N. and Now{\'e}, Ann and Bazzan, Ana L. C. and Talbi, El Ghazali and Danoy, Gr{\'e}goire and Silva, Bruno Castro da},
	title = {A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning},
	booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems ({NeurIPS} 2023)},
	year = {2023}
}

8. Change Replace pql.py with the file I uploaded

Replace pql.py in morl_baselines/multi_policy/pareto_q_learning
with the pql.py I uploaded.


## Running ns-3 environment

To run scenario, open the terminal and run the command:

```shell
chmod +x ./Rerun_MOHOLA.sh
./bash Rerun_MOHOLA.sh
```

Note that, you don't have to repeat the following command after your first running.

```shell
chmod +x ./Rerun_MOHOLA..sh
```

If you want to run only one episode, run the command:

```shell
./waf --run scratch/NS3_Env_HetNet.cc
```

## Running agent

In the directory scratch, there are MOHOLA agent file.

open a new terminal and run the command:

```shell
cd ./scratch
python3 MOHOLA_Agent.py
```

Contact
================================

Kihoon Kim, Korea University, rlgns1109@korea.ac.kr

Eunsok Lee, Korea University, tinedge@korea.ac.kr


How to reference MOHOLA?
================================
Please use the following bibtex:

<blank>

