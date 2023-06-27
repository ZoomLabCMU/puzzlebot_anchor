# Reconfigurable Robot Control Using Flexible Coupling Mechanisms
Sha Yi, Katia Sycara, and Zeynep Temel  

<img src="/img/robot_demo.jpg" width="45%" />   <img src="/img/bulletsim_line4.gif" width="45%" />

If you find our work helpful, please consider citing our paper:
```
@inproceedings{yi2023reconfiguration,
  title={Reconfigurable Robot Control Using Flexible Coupling Mechanisms},
  author={Yi, Sha and Sycara, Katia and Temel, Zeynep},
  booktitle={Robotics science and systems},
  year={2023}
}
```

## Installation

1. Create conda environment with 
```
conda env create -f puzzle_env.yml
```
This will create a environment named `puzzle`. You will need to activate it with
```
conda activate puzzle
```

2. Install three other dependencies with `pip` in the `puzzle` environment.
```
pip install casadi
pip install polytope
pip install pybullet
```
3. The package is originally in the ROS catkin source workspace due to my hardware interface. But the simulation is independent of ROS. However, if you do not have ROS installed, you might need to install another dependency:
```
pip install catkin-pkg
```
Then you may install this package with
```
pip install -e .
```

## Generate URDF file
There are multiple PuzzleBot URDFs that can be generated. Currently I'm using the simplified one. Run the following in the package root directory.
```
python scripts/generate_urdf.py 1
```
This should generate `puzzlebot.urdf` in the `urdf/` directory.

## Run Simulation
Run the following for the simulation.
```
python bin/run_sim.py
```
Let me know if you encounter any errors.

## Run on Hardware
Since ROS 1 is based on python 2 while our package uses python 3, the user is suggested to resolve this conflict on their computer based on their local environment. [This link](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) may be helpful.

After resolving the python conflict, you may first move the package to your catkin workspace source directory, e.g. `catkin_ws/src`. Then install this package with 
```
catkin build
source devel/setup.bash
```
You can then launch the package with
```
roslaunch puzzlebot_assembly run_multi.py N:=$NUMBER_OF_ROBOTS
```

### Disclaimer
This code is only tested fully on Ubuntu 18.04 with python version 3.7. It is partially tested on Macbook but the simulation and optimization parameters may need additional tuning.
