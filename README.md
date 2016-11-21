# CNN
This program implements a convolutional neron network, and is applied to solve digit recoginition task (MNIST).

## Note on Windows:
+ The depend library HMatrix requires some efforts to install.
  + The version on Hackage of 0.17.0.2 does not work. It will produce linkage errors, complaining that symbols "random" and "rand48" not found. Instead, please download [the head version](https://github.com/albertoruiz/hmatrix) of HMatrix from github. Place it on the upper directory of this project, as specified in the stack.yaml.
  + HMatrix itself requires openblas to get build. It can be found [here](http://www.openblas.net/) and the binary package of version 0.2.15 works ok.
    + or, if you want to compile latest Openblas your self. You will need get a MSYS2 environment, install the MINGW_W64, etc. HMatrix needs the OpenBlas with LAPACK included, therefore GFORTRAN is also needed.
  + Apart from the binray package of openblas, the **libgfortran-3.dll** is needed to be seen under the **GHC_INSTALL_DIR\mingw\bin**. I find a easy way by starting a msys2 shell (the stack utility already installs one), run _packman -S mingw-w64-x86_64-gcc-libgfortran_, after that you will find a libfortran-3.dll under the MSYS2_INSTALL_DIR\mingw64\bin. Then simply copy it over.
