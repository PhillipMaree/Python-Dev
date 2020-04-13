####Introduction
Projects in this folder uses the optimization development framework, co-simulation framework and development ide
as defined in the docker container based environments created by building the container image for projects in

    https://github.com/PhillipMaree/Docker.git
    
Note that one only need to build the image for *02_intellij_ide* which will create an image called *jpmaree/optimization_ide:latest*.

####Getting started

An interactive terminal environment can be invoked which will mount the projects folder in the 

Each project folder contains a bash script which will mount the respective project folder within the running *jpmaree/optimization_ide:latest* container.

Support has been added to split the terminal screen into multiple screens by using *GNU Screen* functionality. To run the latter
type *screen*.

The user can modify the *GNU Screen* shortcuts and properties by modifying *.screenrc* in the projects root folder.
 
The following *GNU Screen* are helpful:

    1. Split screen vertically: Ctrl-a |
    2. Split screen horizontally: Ctrl-a Shift-s
    3. Start a new command prompt: Ctrl-a c