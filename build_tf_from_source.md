# Build Tensorflow from source

```sh
git clone https://github.com/google/tensorflow.git
cd tensorflow
git checkout r1.12
./configure
```

If needed,
```sh
source activate tf-py36
```


```sh
bazel build --config=mkl --config=cuda //tensorflow/tools/pip_package:build_pip_packag
```

```txt
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: Y

Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N

Do you wish to build TensorFlow with ROCm support? [y/N]: N

Do you wish to build TensorFlow with CUDA support? [y/N]: Y

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0

Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda-10.0

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.3.1

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0

Do you wish to build TensorFlow with TensorRT support? [y/N]: N

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3.5

Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 5.0] 5.0

Do you want to use clang as CUDA compiler? [y/N]: N

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc

Do you wish to build TensorFlow with MPI support? [y/N]: N

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=native

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:N
--------------------------------------------------------------------

Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 9694.575s, Critical Path: 318.16s
INFO: 14328 processes: 14328 local.
INFO: Build completed successfully, 17818 total actions
```

```sh
bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
```

```txt
Sat Apr 27 22:44:12 UTC 2019 : === Preparing sources in dir: /tmp/tmp.fY2JF86G7M
~/git/tensorflow ~/git/tensorflow
~/git/tensorflow
Sat Apr 27 22:44:21 UTC 2019 : === Building wheel
warning: no files found matching '*.pd' under directory '*'
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
warning: no files found matching '*.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '*' under directory 'tensorflow/include/Eigen'
warning: no files found matching '*.h' under directory 'tensorflow/include/google'
warning: no files found matching '*' under directory 'tensorflow/include/third_party'
warning: no files found matching '*' under directory 'tensorflow/include/unsupported'
Sat Apr 27 22:44:57 UTC 2019 : === Output wheel file is in: /home/pydemia/git/tensorflow/tensorflow_pkg
```

```sh
cd tensorflow_pkg
pip install tensorflow-1.12.2-cp36-cp36m-linux_x86_64.whl
```sh
