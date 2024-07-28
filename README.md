OS and software preparation:
We base our experiment environment on Ubuntu 20.04 LTS and highly recommend that you do the same. This streamlines the setup process and avoids unexpected issues cause by incompatible software versions etc. Please make sure that you have Python installed. Also make sure that you have root or sudo permission.

This branch contains the entire ns-3 network simulator (ns-3.33) with ns3-gym (opengym) module.

Install ns-3 and dependencies
The first part of the preparation is to clone the repository:
git clone https://github.com/tinedge/SLC2.git
Next, install all dependencies required by ns-3.
apt-get install gcc g++ python python3-pip
Install ZMQ and Protocol Buffers libs:
sudo apt-get update
apt-get install libzmq5 libzmq3-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
Install Pytorch.
pip3 install torch
Following guideline of installation in https://pytorch.org

Building ns-3
chmod +x ./waf
./waf configure
./waf build
Install ns3-gym
pip3 install --user ./src/opengym/model/ns3gym
Running ns-3 environment
To run small scale scenario, open the terminal and run the command:

chmod +x ./Rerun_MOHOLA.sh
./bash Rerun_MOHOLA.sh
Note that, you don't have to repeat the following command after your first running.

chmod +x ./Rerun_MOHOLA..sh
If you want to run only one episode, run the command:

./waf --run scratch/NS3_Env_small.cc
To run large scale scenario, open the terminal and run the command:


Running agent
In the directory scratch, there are SLC2 agent files for small and large scale scenarios.

For small scale scenario, open a new terminal and run the command:

cd ./scratch
python3 SLC2_Agent_small.py
For large scale scenario, open a new terminal and run the command:

cd ./scratch
python3 SLC2_Agent_large.py
Contact

Kihoon Kim, Korea University, rlgns1109@korea.ac.kr

Eunsok Lee, Korea University, tinedge@korea.ac.kr


How to reference MOHOLA?
Please use the following bibtex:
