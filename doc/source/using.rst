==============
Using the code
==============

The requirements are:

* `Python <http://www.python.org/>`_ 2.6 or higher
* `numpy <http://www.numpy.org/>`_ and `scipy <http://www.scipy.org/>`_

These packages can be installed by a single command from Linux terminal::

    $pip install -r requirements.txt

**requirements.txt** is contained in the root directory of this project.

**Before installing these packages, make sure you have lapack or mkl library on your host**.
It is always recommended to use `Anaconda <https://www.continuum.io/downloads/>`_ to install these packages.

Download the code using the `Download ZIP
<https://github.com/GiggleLiu/Jacobi_Davidson/archive/v1.0.tar.gz>`_
button on github, or run the following command from a terminal::

    $ wget -O Jacobi_Davidson-1.0.tar.gz https://github.com/GiggleLiu/Jacobi_Davidson/archive/v1.0.tar.gz

Within a terminal, execute the following to unpack the code::

    $ tar -xvf Jacobi_Davidson-1.0.tar.gz
    $ cd Jacobi_Davidson-1.0/source/

Once the relevant software is installed, each program is contained
entirely in a single file.  A sample file source/sample.py, for instance, can be
run by issuing::

    $ python sample.py
