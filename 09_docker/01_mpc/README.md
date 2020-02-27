# To build

#### Build arguments and launch
The following arguments can be used during building:

Usage:
    
    --build-arg ARG=[argument]
    
Supported ARG types:
    
    COINHSL_SRC_PATH  [<string>] path of coinhsl with /include as base
    CPLEX_SRC_PATH [<string>] path to cplex binary to be installed
    
To build, call i.e.:
    
    docker build --rm -f Dockerfile --build-arg COINHSL_SRC_PATH=./coinhsl/coinhsl-2019.05.21.tar.gz  --build-arg CPLEX_SRC_PATH=./cplex/COSCE129LIN64.bin -t sintef_dev .
    
To launch the container and mount your projects folder, run:

    export PROJECTS_DIR=
    docker run --detach --name sintef_mpc_dev_env --volume $PROJECTS_DIR:/projects


####Troubleshooting
1. If you cannot delpoy due to HTTP issues, then you need to re-authenticate dcos via:

	docs auth login

2. See SSH config file at:

	~/.ssh/config

3. Remove all docker images:

	docker rmi -f $(docker images -a -q)

4. Remove all docker containers:

	docker rm -vf $(docker ps -a -q)
