===================
Tutorial
===================

Example
-----------------------
First, I will show a rather simple example, it looks like:

.. literalinclude:: ../../source/sample.py
   :linenos:

The output looks like

.. literalinclude:: ../../source/output.demo

A lot of time consummed in calculating the approximate K = (A-sigma*M)^-1,
But with proper K, we find the eigenvalues converge extremely fast!

Advanced Parameters can be found in API attached behind,
like `projector` allows us to project the state into desired space which can be used to remove degeneracy.
