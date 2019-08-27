# Real-Time Weapons and Gun Detection

A neural network project for detecting weapons and guns from live video feeds.

![realtime weapons and gun detection by james burnett burnett](https://jroburnett.com/wp-content/uploads/2019/08/real-time-weapons-gun-detection.gif)

The focus of this project is to create a ConvNet (convolutional neural network) model for detecting guns and weapons in live video feeds. This is a dlib c++ project with a CNN model based on the dlib examples, but simplified into two programs, on for training the network and another for testing the netwiork.

The goal is to find the best possible CNN model for detecting handguns from video frames.

More at https://burnett.tech

## RequireMents
  - Ubuntu 18.04 (not nessessarly required but easiest to get going)
  - NVidia CUDA 10 installed with driver for Xorg
  - cudNN 7.3 for CUDA 10 support 
  - dlib 19.16
  - opencv2
  - build-essential and cmake for Ubuntu 18.04

## Getting Started
  - First, install CUDA 10 from Ubuntu. use the local .deb installer as it will install various dependencies for your system. Make sure you also install the driver. 
  - Next, download and install the cudNN 7.3 deb file (local) dpkg -i nvidi-cudnn-7.3.x.x.x.deb etc....
  - After you are finished install the NVidia requirements you will need to install a compiler and cmake on Ubuntu with 'sudo apt-get install build-essential cmake'
  - You also need to install the opencv development tools. 
  - The final step is to install dlib, which you can follow the instructions on the dlib.net webpage.


### Installing compilers and cmake
```
$ apt-get install build-essential cmake
```

### Installing OpenCV 
```
$ apt-get install libopencv-dev libopencv-core-dev

```

### Compiling dlib as a shared library.
Make sure to compile dlib as a shared library. 

```
$ cd dlib-19.16
$ cmake -DBUILD_SHARED_LIBS=1 ..
$ make
$ sudo make install
 
```

### Compile and install gun_trainer and gun_tester
Once dlib is built and install all you have to do is build the gun_trainer and gun_tester programs. 

```
$ cd realtime-gun-detection
$ mkdir build; cd build
$ cmake ..
$ make

```



### Pyhton tools
There are some SVM based object detection tools located in the python_object_detection folder. These tools are also based on the SVM tools from dlib. You will need the opencv2 python modules and dlib python modules for these tools. See dlib.net.
