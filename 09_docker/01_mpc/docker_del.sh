#!/bin/bash
docker rm -vf $(docker ps -a -q)      # remove all containers
docker rmi -f $(docker images -a -q)  # remove all images