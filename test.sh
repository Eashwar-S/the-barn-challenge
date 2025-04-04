#!/bin/bash
for i in {0..49} ; do
    n=`expr $i \* 6` # 50 test BARN worlds with equal spacing indices: [0, 6, 12, ..., 294]
        for j in {1..10} ; do            
            # run the test
            python run_predictive_dwa.py --world_idx $n
            sleep 5
        done
done