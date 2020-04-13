#!/bin/bash
echo -e '\n \e[1;34mMounted project base directory:\033[0m\e[1;33m'$PWD'\033[0m\n'
docker run -it --privileged --volume=$HOME"/.Xauthority:/root/.Xauthority:rw" --mount type=bind,source=$PWD,target=/home/$USER --env="DISPLAY" --net=host jpmaree/optimization_ide:latest

