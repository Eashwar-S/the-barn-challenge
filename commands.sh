#!/bin/bash
docker build -t barn_simulation_modified .

# Allow X server connections from Docker
xhost +local:docker

# Run the Docker container
docker run -it --rm \
  --net=host \
  --env DISPLAY=$DISPLAY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume /dev/shm:/dev/shm \
  --device /dev/dri \
  --name barn_container \
  barn_simulation_modified:latest
