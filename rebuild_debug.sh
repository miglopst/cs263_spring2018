#bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel build --config=cuda --compilation_mode=dbg --strip=never //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall tensorflow
pip install /tmp/tensorflow_pkg/tensorflow*.whl
