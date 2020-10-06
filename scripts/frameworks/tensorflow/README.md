# Tensorflow

![Tensorflow API Levels](tensorflow_programming_environment.png)

## Basics on Tensorflow

* [The Bottom-Line of `tensorflow`](basics/tensorflow_basics.md#the-bottom-line-of-tensorflow)
  - [Tensor]()
  - [Session]()
    - [`tf.Session.run()` vs `Tensor.eval()`]()
  - [Scope]()

* [Basic Objects]()
  - [Data Types]()
    - [Constants]()
    - [Sequences]()
    - [Randoms]()
  
  - [Data Structures]()
    - [Tensor Transformations]()
      - [Casting]()
      - [Shaping]()
      - [Slicing & Joining]()
      - [Casting]()

* [Control Flow]()
  - [Operators]()
    - [Logical Operators]()
    - [Comparison Operators]()
    - [Debugging]()

* [Computation(Math)]()
  - [Arithmetics]()
  - [Basic Functions]()
  - [Matrix Functions]()
  - [Tensor Functions]()
  - [Complex Number Functions]()
  - [Functions for Reduction]()
  - [Scan(total, cumulative)]()
  - [Segmentation]()
  - [Sequence Comparison and Indexing]()



## Intermediates in Tensorflow

* [`tf.layers` Module]()
  - [Data Types]()
  - [Data Structures]()

* [`tf.nn` Module]()
  - [Data Types]()
  - [Data Structures]()

* [`tf.contrib` Module]()
  - [Data Types]()
  - [Data Structures]()

## Advanced in Tensorflow
* [`tf.data`]()
* [`tf.distribute`]()

---

### Installation from source (for diff CUDA version)

```sh
cd
git clone https://github.com/tensorflow/tensorflow
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

cd ~/Downloads
wget https://github.com/bazelbuild/bazel/releases/download/0.16.1/bazel-0.16.1-installer-linux-x86_64.sh
sudo bash ./bazel-0.16.1-installer-linux-x86_64.sh --prefix=/usr
source ~/.bashrc

source activate tf-py36
cd git/tensorflow
./configure

```

```sh
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.16.1 installed.
Please specify the location of python. [Default is /usr/share/anaconda3/envs/tf-py36/bin/python]:


Found possible Python library paths:
  /usr/share/anaconda3/envs/tf-py36/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/usr/share/anaconda3/envs/tf-py36/lib/python3.6/site-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]:
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]:
Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]:
Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]:
Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]:
Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]:
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]:
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]:
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]:
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.2


Please specify the location where CUDA 9.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.0


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Do you wish to build TensorFlow with TensorRT support? [y/N]: N
No TensorRT support will be enabled for TensorFlow.

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 1.3


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.7,3.7]:


Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```


`gcc > 5`:

```sh
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"

```


## tensorflow & cuda-10.0


ubuntu 16.04
```sh
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-410
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update && \
        sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0

```
