# SE2Wave #

### What is this repository for? ###

* Provides a toy implementation of the spectral element method in 2D for wave propagation

### How do I get set up? ###

* Requires PETSc (`https://www.mcs.anl.gov/petsc`)
* Must be compiled with PETSc version 3.9. This can be downloaded from here `https://www.mcs.anl.gov/petsc/download`

### Common options ###

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

### Who do I talk to? ###

* Email: dave.mayhem23@gmail.com
* Via BitBucket: Use `@dmay` in any BB dialogue box
