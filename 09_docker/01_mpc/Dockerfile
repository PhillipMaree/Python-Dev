FROM ubuntu:18.04

MAINTAINER Phillip Maree <phillip.maree@sintef.no>

# working directory setup 

WORKDIR /scratch
ADD /include ./

# build arguments, depending on availibility of thrirdparty software (set build time arguments -build-arg )

ARG USER_NAME
ARG COINHSL_SRC_PATH
ARG CPLEX_SRC_PATH

# environment variables
ENV USER_LOCAL_LIB="/usr/local/lib"
ENV IPOPT_HOME=/opt/Ipopt/build
ENV IPOPT_INC_DIR=/usr/local/include/coin-or

ENV CPLEX_INC_DIR=/opt/Cplex/cplex/include
ENV CPLEX_CFLIB="-L/opt/Cplex/cplex/lib/x86-64_linux/static_pic -lcplex -lpthread -lm"
ENV CPLEX_LIB=/opt/Cplex/cplex/lib/x86-64_linux/static_pic/libcplex.a

ENV BONMIN_INC_DIR=/opt/Bonmin/build/include/coin
ENV BONMIN_LIB=/opt/Bonmin/build/lib/libbonmin.so

# give the user root access

RUN echo "$USER ALL = NOPASSWD: ALL" >> /etc/sudoers \
   && apt-get update && apt-get install -y

# append local lib in system path

RUN  echo $USER_LOCAL_LIB >> /etc/ld.so.conf.d/sintef.conf \
     && ldconfig

# configure tzdata

ENV TZ 'Europe/Amsterdam'

RUN echo $TZ > /etc/timezone && \
  apt-get update && apt-get install -y tzdata && \
  rm /etc/localtime && \
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
  dpkg-reconfigure -f noninteractive tzdata && \
  apt-get clean

# install required packages

RUN apt-get install -y \
   wget \
   swig \
   git \
   pkg-config \
   apt-utils \
   libblas-dev \
   liblapack-dev \
   gcc \
   g++ \
   gfortran patch \
   default-jre \
   default-jdk \
   subversion \
   nano \
   locate \
   cmake \
   --install-recommends

# install python

RUN echo -e '\n \e[1;34mInstall Python \033[0m \n'

RUN apt-get install -y ipython python-dev python-numpy python-scipy python-matplotlib --install-recommends

RUN echo -e '\n \e[1;34mInstall Python 3 \033[0m \n'

RUN apt-get install -y python3-pip python3-dev python3-numpy python3-scipy python3-matplotlib\
   && cd /usr/local/bin \
   && ln -s /usr/bin/python3 python \
   && pip3 install --upgrade pip

# prepare ipopt installation

RUN wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.13.0.tgz \
   && mkdir -p /opt/Ipopt/ && mkdir -p /opt/Ipopt/thirdparty \
   && tar -xvzf Ipopt-3.13.0.tgz -C /opt/Ipopt/ --strip-components 1

# install thirdparty software for ipopt
RUN echo -e '\n \e[1;34mInstall Ipopt thirdparty libraries\033[0m \n'

RUN git clone https://github.com/coin-or-tools/ThirdParty-ASL.git /opt/Ipopt/thirdparty/ASL \
   && cd /opt/Ipopt/thirdparty/ASL &&  ./get.ASL \
   && ./configure && make && make install && cd -

RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git /opt/Ipopt/thirdparty/Mumps \
   && cd /opt/Ipopt/thirdparty/Mumps && ./get.Mumps \
   && ./configure && make && make install && cd -

RUN git clone https://github.com/coin-or-tools/ThirdParty-Metis.git /opt/Ipopt/thirdparty/Metis \
   && cd /opt/Ipopt/thirdparty/Metis && ./get.Metis \
   && ./configure && make && make install && cd -

RUN if [ -r  "$WORKDIR/$COINHSL_SRC_PATH" ] ; then echo -e '\n \e[1;31mHSL not availible \033[0m\n' ; else \
        echo -e '\n \e[1;32mHSL thirdparty libraries provided\033[0m \n' ; \
        git clone https://github.com/coin-or-tools/ThirdParty-HSL.git /opt/Ipopt/thirdparty/HSL ; \
        mkdir -p /opt/Ipopt/thirdparty/HSL/coinhsl &&  tar -xvzf ${COINHSL_SRC_PATH} -C /opt/Ipopt/thirdparty/HSL/coinhsl --strip-components 1 ; \
        cd /opt/Ipopt/thirdparty/HSL && ./get.HSL ; \
        ./configure && make && make install ; \
    fi

# install ipopt

RUN echo -e '\n \e[1;34mInstall Ipopt\033[0m \n'

RUN cd /opt/Ipopt/ && mkdir build && cd build \
   && ../configure --prefix=/usr/local ADD_FFLAGS=-fPIC ADD_CFLAGS=-fPIC ADD_CXXFLAGS=-fPIC --disable-java \
   && make && make install && make test \
   && cd  /usr/local/include/ && cp -r coin-or/*.* .

# install cplex (optional)
RUN echo -e '\n \e[1;34mInstall Cplex\033[0m \n'

RUN if [ -r  "$WORKDIR/$CPLEX_SRC_PATH" ] ; then echo -e '\n \e[1;31mCPLEX libraries not provided\033[0m \n' ; else \
       echo -e '\n \e[1;32mCPLEX libraries provided\033[0m \n'; \
       chmod +x ${CPLEX_SRC_PATH} &&  ./${CPLEX_SRC_PATH} -f  cplex.properties; \
    fi

# install bonmin

RUN echo -e '\n \e[1;34mInstall Bonmin\033[0m \n'

RUN wget https://www.coin-or.org/download/source/Bonmin/Bonmin-1.8.8.tgz \
   && mkdir -p /opt/Bonmin/ && tar -xvzf Bonmin-1.8.8.tgz -C /opt/Bonmin/ --strip-components 1 \
   && cd /opt/Bonmin/ && mkdir build && cd build \
   if [ -r  "$WORKDIR/$CPLEX_SRC_PATH" ] ; then \
      echo -e '\n \e[1;32mConfigure BONMIN without CPLEX libraries \033[0m \n' ; ../configure --with-ipopt-datadir=${IPOPT_HOME} --with-ipopt-incdir=${IPOPT_INC_DIR} ; else \
      echo -e '\n \e[1;32mConfigure BONMIN with CPLEX libraries \033[0m \n' ;../configure --with-ipopt-datadir=${IPOPT_HOME} --with-ipopt-incdir=${IPOPT_INC_DIR} --with-cplexlib=${CPLEX_CFLIB} --with-cplexincdir=${CPLEX_INC_DIR}; \
   fi \
   make clean && make

RUN ldconfig -v && cd /opt/Bonmin/build/Bonmin/test && make clean && make test && cd /opt/Bonmin/build && make install

# install casadi

RUN echo -e '\n \e[1;34mInstall Casadi\033[0m \n'

RUN git clone https://github.com/casadi/casadi.git -b master /opt/Casadi \
   && cd /opt/Casadi && mkdir build && cd build \
   && cmake -DWITH_PYTHON=ON \
   -DWITH_PYTHON3=ON \
   -DWITH_IPOPT=ON \
   -DWITH_BONMIN=ON \
   -DWITH_CPLEX=ON \
   -DBONMIN_LIBRARIES=$BONMIN_LIB \
   -DBONMIN_INCLUDE_DIRS=$BONMIN_INC_DIR \
   -DCPLEX_LIBRARY=$CPLEX_LIB \
   -DCPLEX_INCLUDE_DIR=$CPLEX_INC_DIR ..

RUN cd /opt/Casadi/build && make && make install

# create user working directory

RUN useradd -ms /bin/bash $USER_NAME
USER $USER_NAME
WORKDIR /home/$USER_NAME



