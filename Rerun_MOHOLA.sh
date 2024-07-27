
#!/bin/bash
for i in {1..20000}
do  
    rm -rf scratch/wandb
    ./waf --run "scratch/NS3_Env_HetNet.cc --RunNum=$(($i))"
done