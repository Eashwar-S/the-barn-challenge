#!/bin/bash
for i in {300..359} ; do
    for j in {1..10} ; do            
        # run the test
        python run_predictive_dwa.py --world_idx $i
        sleep 5
    done
done