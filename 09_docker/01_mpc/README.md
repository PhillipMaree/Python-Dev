#### Integrated development environment for optimization
This project entails a dockerfile formulation that encapsulates various solvers (both open source and propriety software)
to be used to solver optimization problems. The target application is to be able to formulate Model Predictive Control 
problem that can among other either be NLP, MINLP, MILP etc of nature.

The following solvers have been included in the dockerfile image:
    
 1. **Ipopt** (build with **HSL** libraries if supplied)
 2. **Bonmin** (build with **CPLEX** for MIQP, if supplied)
 3. **Casadi** (interfaces enabled are Ipopt, Bonmin, Python, Cplex)
 
 #####Pull latest build docker image
 To pull a pre-build image, execute:
 
    docker image pull jpmaree/sintef-digital:latest
 
 #####Push latest build docker image (require privileges)
 Login to docker:
 
    docker login
    
 Push latest build image
 
    docker push jpmaree/sintef-digital:tagname
 
 #####Build docker image
 The build the image from scratch, call:
 
    ./docker_build.sh
    
The build.sh should be modified, based on availability of third-party libraries. By using the following flag,
 
    --build-arg ARG=[argument]
    
one can supply values to the following arguments:
    
    COINHSL_SRC_PATH  [<string>] path of coinhsl with /include as base
    CPLEX_SRC_PATH [<string>] path to cplex binary to be installed
    
 #####Run docker container
  The docker container will share the host's XServer to display GUI (ie., matplotlib.pyplot visualization) in the 
  foreground of the host pc. The following command can be executed to run the docker container:
  
    ./dokcker_run.sh [options] 
    
    Options:
    
    -w <string:path> [work directory path to be mounted in countainer at /home/$USER/string:path]
    
As default, the /work folder will be mounted if no options are specified 
    
 #####Purge docker containers and images (!!WARNING!!)
 All images and containers can be purged running the following command:
 
    ./docker_del.sh
 
