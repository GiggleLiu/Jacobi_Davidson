# Jacobi Davison Iterative Eigenvalue Solver
The Jacobi Davison diagonalization, python interface with fortran underlying(planning to do so).

## Install
Dependency

* numpy
* scipy

To install,

```bash
    git clone https://github.com/GiggleLiu/Jacobi_Davison.git
    cd Jacobi_Davidson/source
    python setup.py install
```
## Usage and Documentation
To use the code, build the documents first, e.g. the pdf version can be compiled using
```bash
    cd Jacobi_Davidson/doc
    make latexpdf
    evince build/latex/PyDavidson.pdf
```
Alternatively, you may refer sample file *Jacobi_Davidson/souce/sample.py* for a quick start!
