#!/bin/bash
echo need 2 args, usage: bash run.sh rank num_nodes 
rank=$1
Nnodes=$2
ip=1
if [ "$#" -ne 3 ]
then
  echo python main.py --rank ${rank}  --num_nodes ${Nnodes} --master_ip "10.10.1.${ip}"
  python main.py --rank ${rank}  --num_nodes ${Nnodes} --master_ip "10.10.1.${ip}"
else
  echo python main.py --rank ${rank}  --num_nodes ${Nnodes} --master_ip "10.10.1.${ip}" --outputfile ${3}
  python main.py --rank ${rank}  --num_nodes ${Nnodes} --master_ip "10.10.1.${ip}" --outputfile ${3}
fi

