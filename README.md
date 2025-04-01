<p align="center">
  <img width = "100%" src='res/BARN_Challenge.png' />
  </p>

--------------------------------------------------------------------------------

# ICRA BARN Navigation Challenge

# Submission - Model-based Predictive DWA planner:

# Requirements
If you run it on a local machine without containers:
* ROS version at melodic
* CMake version at least 3.0.2
* Python version at least 3.6
* Python packages: defusedxml, rospkg, netifaces, numpy, scikit-learn, scipy, filterpy, matplotlib
* Docker 

# Installation
## Due to the difficulties faced in running singularity containers, we have provided Dockerfile to build a docker image and run it.

Follow the instruction below to run simulations in Docker containers.

1. Clone this repo
```
https://github.com/Eashwar-S/the-barn-challenge.git
cd the-barn-challenge
```

3. Build Docker image and run it:
```
chmod +x commands.sh
./commands.sh
```

## Run Simulations
Run the simulation using the following command:
```
python run_predictive_dwa.py --world_idx 0
```


## Submission
Submitted github link to this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSfZLMVluXE-HWnV9lNP00LuBi3e9HFOeLi30p9tsHUViWpqrA/viewform). Please use the dockerfile provided which follows the icra 2022 guidelines to build and test the planner.