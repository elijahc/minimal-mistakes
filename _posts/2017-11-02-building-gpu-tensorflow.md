---
layout: single
title: Building Tensorflow from source with GPU bindings
---

## Why you should care

Using Tensorflow with a GPU can significantly speed up training and testing time with large models.  
However, just because you finished installing that fancy new TITANX doesn't mean you're ready to go.

I learned the hard way if you don't compile tensorflow from source with gpu drivers it just kinda sits there with nothing to do.

So, with that being said, lets put that GPU to work.

PSA: Full disclosure, I myself did this a long time ago and forgot many of the steps I actually took so I heavily "borrowed" a lot from [this guide](https://gist.github.com/Brainiarc7/6d6c3f23ea057775b72c52817759b25c) which jogged a lot of \(sometimes bad\) memories.

## Getting Started

### **Step 0. Make sure NVIDIA Drivers are uptodate:**

This was a _major headache_ for me, so much so that I'll probably be writing up something specifically about upgrading nvidia drivers "cleanly".

[This guide](https://medium.com/@ikekramer/installing-cuda-8-0-and-cudnn-5-1-on-ubuntu-16-04-6b9f284f6e77)  
and others like it were helpful when I was trying to figure it all out.

---

**Step 1. Install NVIDIA CUDA:**

To use TensorFlow with NVIDIA GPUs, the first step is to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

**Step 2. Install NVIDIA cuDNN:**

Once the CUDA Toolkit is installed, [download cuDNN v5.1 Library for Linux](https://developer.nvidia.com/rdp/cudnn-download#a-collapseTwo)  
\(note that you will need to register for the  
  [Accelerated Computing Developer Program](https://developer.nvidia.com/accelerated-computing-developer)\).

Once downloaded, uncompress the files and copy them into the CUDA Toolkit directory \(assumed here to be in /usr/local/cuda/\):

```bash
$ sudo tar -xvzf cudnn-8.0-* -C
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

**Step 3. Install and upgrade PIP:**

In order to compile python binaries you'll need the python-dev.

```bash
$ sudo apt-get install python-dev
```

To use our freshly compiled tensorflow python binaries we'll need to install them with pip.  If you [installed pyenv to manage your python versions](https://jzlab.gitbooks.io/jzkb/content/pyenv.html) you should already have pip as a part of one of your python installations. Just remember to set which python version/environment you want gpu-tensorflow to be installed on.

```bash
$ pyenv local <python version>
```

If you aren't using pyenv \(you probably should be...\), at least make sure you have pip installed.

> This is just one example of installing pip via the package manager but there are lots of ways to do it.

```bash
$ sudo apt-get install python-pip
```

and make sure its updated

```bash
$ pip install --upgrade pip
```

**Step 4. Install Bazel** - Google's monolith of a build tool:

To build TensorFlow from source, the Bazel build system must first be installed as follows.

```bash
$ sudo apt-get install software-properties-common swig
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
$ echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install bazel
```

**Step 5. Fetch Tensorflow Source**

At least for now, Tensorflow has to be compiled from source to take advantage of GPU acceleration.

First, clone the TensorFlow source code repository:

```bash
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git reset --hard a23f5d7
```

Then run the configure script as follows \(example output\):

```bash
$ ./configure

Output:

    Please specify the location of python. [Default is /usr/bin/python]: [enter]
    Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
    No Google Cloud Platform support will be enabled for TensorFlow
    Do you wish to build TensorFlow with GPU support? [y/N] y
    GPU support will be enabled for TensorFlow
    Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]: [enter]
    Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
    Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: [enter]
    Please specify the Cudnn version you want to use. [Leave empty to use system default]: 5
    Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: [enter]
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size.
    [Default is: "3.5,5.2"]: 5.2,6.1 [see https://developer.nvidia.com/cuda-gpus]
    Setting up Cuda include
    Setting up Cuda lib64
    Setting up Cuda bin
    Setting up Cuda nvvm
    Setting up CUPTI include
    Setting up CUPTI lib64
    Configuration finished
```

Then call bazel to compile TensorFlow and get ready to build the pip package:

```bash
$ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

And finally, build the pip package!

```bash
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

This will build the package with optimizations for FMA, AVX and SSE \(common CPU acceleration routines\) as well as cuda bindings

---

Finally, use pip to install the source compiled tensorflow package.

If using pyenv:

```bash
$ cd /tmp/tensorflow_pkg/

# Pick which python envinronment to install gpu-tensorflow into
$ pyenv local 3.5.3

# Confirm you've selected the python envorinment you want
$ pyenv which pip
/home/elijahc/.pyenv/versions/3.5.3/bin/pip

$ pip install --upgrade tensorflow-*.whl
```

If not pyenv:

```bash
$ sudo pip install --upgrade /tmp/tensorflow_pkg/tensorflow-*.whl
```

**Step 6. Upgrade protobuf:**

Upgrade to the latest version of the protobuf package:

For Python 2.7:

```bash
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl
```

For Python 3.4:

```bash
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp34-none-linux_x86_64.whl
```

**Step 6. Test your installation:**

To test the installation, open an interactive Python shell:

```bash
$ cd ~/
$ ipython
```

> **NOTE:** Changing directories is important, if you run an interactive shell from where you installed tensorflow it will likely pick up the local version in that directory thereby possibly giving you a false positive result \(like it did to me\)

and import the TensorFlow module:

```python
import tensorflow as tf
print(tf.__version__)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

You should see tensorflow's version which should match the version you checked out from source and “Hello, TensorFlow!”.

Happy training!

-edc
