# Exercises 1 -  Extended Kalman Filter

This repository contains some runnable python files, and a single ROS package with two launch files. They're indipendent.

You can read the report [here](report.pdf)

## Running the python code

The EKF code for the **single pendulum** and its original main file can be run with
    
    python3 main.py

If you wish to run the **double pendulum** (DP) run this instead

    python3 main_dp.py

## ROS Package installation guide

Clone the repository wherever you like, (this tutorial assumes it's the `~/Downloads` folder), and move the `ekf` folder on your `catkin_ws` directory

These instructions assume you haven't changed your standard `catkin_ws` folder location

1. `cd  ~/Downloads/`
1. `git clone https://github.com/PolitoVandal/exercise1-ekf-enfff.git`
1. `cd exercise1-ekf-enfff`
1. Move the ROS package inside the catkin folder `mv ekf/ ~/catkin_ws/src`
1. Give running privileges to the python code `chmod +x ~/caktin_ws/src/ekf/*.py`
1. `cd  ~/catkin_ws && catkin make`
1. Restart your terminal or source your setup file with `source ~/catkin_ws/devel/setup.bash`


This ROS package will contain two launch files, one for the single pendulum and the other for the double pendulum

## Running the ROS Package

The simplest way is to run it using the launch file I created, to do so

    roslaunch ekf single_pendulum.launch

or, equivalently,

    roslaunch ekf double_pendulum.launch

The more complex alternative is to open up **five different terminals** and run separately these commands

    roscore
    rosrun ekf pendulum.py
    rosrun ekf ekf_singlependulum.py
    rosrun ekf sensor.py
    rqt_plot

In any case, once the rqt_plot window opens up, make sure to show *at least* the topics `real_system/x` and `estimated_system/x`, and change the plot settings to show the graph in all its glory

Here's an example of what you should see

![This should be a ROS rqt_plot example](images/ros_plot_example.gif)