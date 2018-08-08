## How to run
---------------------
> 1. 如果是第一次运行，请使用`/.../tensorlow/CNN_c/OpenBLAS_install.sh`安装OpenBLAS(需要root权限)
> 2. 在执行测试程序run_test.py的终端里输入`$ export OPENBLAS_NUM_THREADS=4`.
> (或者直接将`export OPENBLAS_NUM_THREADS=4`写入到 ~/.bashrc)
> 3. 编译c++部分, 在/.../tensorlow/CNN_c中运行 `$ make`
> 4. 运行`$ python ./run_test.py tensorlow`
