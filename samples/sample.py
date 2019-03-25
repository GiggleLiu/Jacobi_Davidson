from numpy import random
from scipy.linalg import norm, eigh
from scipy import sparse as sps
import time

from pydavidson import JDh

# set up constants
N, N1 = 1000, 10  # matrix dimension, non-zero elements each row.
sigma = 0
k = 3

# Construnct a Random Hermitian Matrix
mat = sps.coo_matrix((random.random(N1 * N) + 1j * random.random(N1 * N),
                      (random.randint(0, N, N1 * N), random.randint(0, N, N1 * N))), shape=(N, N))
mat = mat.T.conj() + mat + sps.diags(random.random(N), 0) * 3

# solve it!
e, v = JDh(mat, k=k, v0=None, M=None,  # calculate k eigenvalues for mat, with no initial guess.
           # set the tolerence and maximum iteration as a stop criteria.
           tol=1e-12, maxiter=1000,
           # calculate selected(SL) region near sigma.
           which='SL', sigma=sigma,
           # set up the solver for linear equation A*x=b(here, Jacobi-Davison correction function),
           # we use bicgstab(faster but less accurate than '(l)gmres') here.
           # preconditioning is used during solving this Jacobi-Davison correction function.
           linear_solver_maxiter=20, linear_solver='bicgstab', linear_solver_precon=True,
           iprint=1)  # medium information are printed to stdout, see advanced parameters in API.

print('Get eigenvalues %s' % e)
