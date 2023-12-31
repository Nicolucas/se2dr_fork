## SE2Wave ##

`SE2Wave` is a toy implementation of the spectral element method in 2D for wave propagation problems in heterogenous media.

#### Features 
* Run-time selection of polynomial degree
* Extensible source-time function object
* Multiple sources, which independent source-time functions
* Piece-wise constant (cell-wise) material properties
* Support for MPI (parallel) execution


#### Implementation restrictions
* Rectangular domains and rectangular elements only
* Support for multiple receivers is not generalised (hard coded output) or optimized (no buffering of output)
* No PML support


### 1. Installation ###
* Requires a C compiler
* Requires PETSc (`https://www.mcs.anl.gov/petsc`). PETSc can be downloaded from here `https://www.mcs.anl.gov/petsc/download`.
* `se2wave` be compiled with PETSc version 3.12. 
* Once PETSc has been built, the executable `se2wave.app` can be compiled with the command
``` 
make all PETSC_DIR=/path/to/petsc PETSC_ARCH=name-of-petsc-arch
```


### 2. Usage ###

Options are provided as command line arguments

* `-mx` : number of elements in x-direction 
* `-my` : number of elements in y-direction 
* `-bdegree` : degree of polynomial used for spectral element basis function (default 2)
* `-tmax` : maximum simulation time
* `-dt` : time-step size (default will use 0.2 CFL)
* `-nt` : number of time-steps to perform
* `-of` : output frequency

The above options are all optional.

Example command line option:
```./se2wave.app -mx 64 -my 64 -bdegree 6 -tmax 0.4 -nt 100000 -of 100```

Plot the result in gnuplot
```
gnuplot> plot "closestqpsource-receiverCP-uva-64x64-p6-rank0.dat" u 1:5 w l
```
The `vy` component of velocity should like this:

![se2wave-demo-vy](docs/figs/se2wave-demo-vy.png)

### 3. Bug reporting / feature requests ###

* Email: dave.mayhem23@gmail.com
* Communication via BitBucket: Use `@dmay` in any BitBucket dialogue box.
* When reporting a bug, please provide: the branch name you are using (obtained from the command `git branch`); the **complete** stack trace reported by PETSc; any command line options required to reproduce your error.
