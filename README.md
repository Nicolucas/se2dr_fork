# SE2Wave #

### 1. What is this repository for? ###

* Provides a toy implementation of the spectral element method in 2D for wave propagation

#### Current design limitations ####
1. No support for MPI (parallel) execution
2. Support for multiple receivers is not ideal (no buffering, hardcoded output)

### 2. How do I get set up? ###

* Requires a C compiler
* Requires PETSc (`https://www.mcs.anl.gov/petsc`)
* Must be compiled with PETSc version 3.9. This can be downloaded from here `https://www.mcs.anl.gov/petsc/download`
* Once PETSc has been compiled, `se2wave` can be compiled simply by executing the following command
``` 
make all PETSC_DIR=/path/to/petsc PETSC_ARCH=name-of-petsc-arch
```

### 3. Common options ###

Options are provided as command line arguments

* `-mx` : number of elements in x-direction 
* `-my` : number of elements in y-direction 
* `-border` : degree of polynomial used for spectral element basis function (default 2)
* `-tmax` : maximum simulation time
* `-dt` : time-step size (default will use 0.2 CFL)
* `-nt` : number of time-steps to perform
* `-of` : output frequency

The above options are all optional.

Example command line option:
```./se2wave -mx 64 -my 64 -border 6 -tmax 0.4 -nt 100000 -of 100```

### 4. Who do I talk to? ###

* Email: dave.mayhem23@gmail.com
* Communication via BitBucket: Use `@dmay` in any BitBucket dialogue box.
* When reporting a bug, please provide: the branch name you are using (obtained from the command `git branch`); the **complete** stack trace reported by PETSc; any command line options required to reproduce your error.
