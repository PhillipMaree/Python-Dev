#!/bin/bash

# default work directory

export WORK_DIR=$PWD"/work"

# process options

while test $# -gt 0; do
  case "$1" in
    -w)
      shift
      if test $# -gt 0; then
        export WORK_DIR=$1
        shift
      else
        echo -e "\e[1;31mNo directory path to process. Usage ./docker_run.sh -w path\033[0m"
        exit 1
      fi
      ;;
    esac
done

# run docker container sharing host's XServer

docker run -it --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume=$WORK_DIR":/home/"$USER"/mnt" --env="DISPLAY" --net=host sintef_dev