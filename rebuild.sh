bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-1.8.0rc0-cp27-cp27mu-linux_x86_64.whl
cd /home/penggu/workspace/tf_example/tutorials/mnist/
