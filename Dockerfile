# Use Ubuntu 18.04 as the base image
FROM ubuntu:18.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=melodic \
    ROS_VERSION=1

# Update package list and install dependencies
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    gnupg2 \
    software-properties-common \
    sudo

# Add ROS repository and install ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-melodic-desktop-full

# Install ROS dependencies and initialize rosdep
RUN apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential && \
    rosdep init && \
    rosdep update

# Install Jackal simulation packages
RUN apt-get install -y ros-melodic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-gazebo

# Install additional dependencies for BARN
RUN apt-get install -y ros-melodic-map-server ros-melodic-navigation

# Set up environment
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Ensure Python3 is properly installed
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip

# Create the virtual environment
RUN python3 -m venv /root/nav_challenge
ENV PATH="/root/nav_challenge/bin:$PATH"

# Install Python dependencies
RUN pip3 install defusedxml rospkg netifaces numpy scikit-learn scipy filterpy matplotlib

RUN apt install python3-tk

# Create ROS workspace and clone repositories
WORKDIR /root/jackal_ws/src

RUN git clone https://github.com/Eashwar-S/the-barn-challenge.git && \
    git clone https://github.com/jackal/jackal.git --branch melodic-devel && \
    git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel && \
    git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel && \
    git clone https://github.com/utexas-bwi/eband_local_planner.git

# Check if rosdep is already initialized before running
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && rosdep update


# Set working directory to ROS workspace
WORKDIR /root/jackal_ws

# Install missing dependencies manually before rosdep install
RUN apt-get update && apt-get install -y \
    ros-melodic-gmapping \
    ros-melodic-sick-tim \
    ros-melodic-rosdoc-lite

# Install ROS dependencies (without using "source" in Docker)
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && rosdep install -y --from-paths . --ignore-src --rosdistro=melodic"

# Build the workspace
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make"


# Set up ROS environment
RUN echo "source /root/jackal_ws/devel/setup.bash" >> ~/.bashrc

# Default command to keep the container running
CMD ["/bin/bash"]

# Instructions for running simulations
RUN echo "To run simulations, use the following commands:" >> /root/README.txt && \
    echo "source ../../devel/setup.sh" >> /root/README.txt && \
    echo "python3 run.py --world_idx 0" >> /root/README.txt

