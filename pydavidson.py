'''
Davidson and Jacobi-Davidson eigenvalue solver.
'''

from numpy import *
from scipy import sparse as sps
from scipy.linalg import eigh
from scipy.linalg.lapack import dsyev
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh,cgs,lgmres,gmres
import pdb,time,warnings

from gs import *

__all__=['davidson','JOCC','JD']

def _normalize(v0):
    '''Calculate the norm.'''
    if ndim(v0)==2:
        return v0/sqrt((multiply(v0,v0.conj())).sum(axis=0))
    elif ndim(v0)==1:
        return v0/norm(v0)

def _eigen_solve(A,method='eigh'):
    '''Solve an eigen-value problem'''
    if method=='eigh':
        return eigh(A)
    elif method=='dsyev':
        w,v,info=dsyev(A)
        return w,v
    else:
        raise Exception('Unknown diagonalization method %s'%method)

def _jdeq_solve(A,Q,r,K=None,shift=0,method='cgs',tol=1e-10):
    '''
    Solve Jacobi-Davidson's correction equation.
        (I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*Q = 0

    Parameters:
        :A: matrix(N,N), the matrix.
        :Q: matrix(N,k), the search space, z is orthogonal to Q.
        :r: 1D array, the distance.
        :K: matrix, preconditioner.
        :method: str, the method to solve linear equation.

    Return:
        1D array, the result z.
    '''
    N=Q.shape[0]
    I=sps.identity(N)
    K=A-shift*I
    QH=matrix(Q).H
    system_matrix=K-Q.dot(QH*K)
    right_hand_side=-r+Q.dot(QH.dot(r))
    if method=='cgs':
        z,info=cgs(system_matrix,right_hand_side,tol=tol)
    elif method=='gmres':
        print system_matrix,right_hand_side
        pdb.set_trace()
        z,info=gmres(system_matrix,right_hand_side,tol=tol)
    elif method=='lgmres':
        z=lgmres(system_matrix,right_hand_side,tol=tol)
    elif method=='symmlq':
        raise NotImplementedError()
    else:
        raise Exception('Unknown method for linear solver %s'%method)
    return z

def JOCC(A,maxiter=1000,tol=1e-12):
    '''
    Jacobi's othorgonal component correction method.

    Parameters:
        :A: matrix, the input matrix, must be symmetric, diagonally dominant and the diagonal entry A[0,0] is the entry with the largest modulus.
        :maxiter: int, the maximum number of iterations.
        :tol: float, the tolerence.
    '''
    #extract parts of matrix
    N=A.shape[0]
    F=A[1:,1:]
    d=F.diagonal()[:,newaxis]
    b=A[1:,0:1]
    alpha=A[0,0]
    #initialisation
    z=zeros((N-1,1))
    lamb_pre=-10086
    for i in xrange(maxiter):
        lamb=alpha+(b.T.conj().dot(z)).real.item()
        z=(multiply(d,z)-F.dot(z)-b)/(d-lamb)
        if abs(lamb-lamb_pre)<tol:
            u=concatenate([[[1]],z],axis=0)
            return lamb,u
        print 'JOCC iter = %s, diff = %s, e = %s'%(i,abs(lamb-lamb_pre),lamb)
        lamb_pre=lamb
    raise Exception('No convergence @JOCC!')

def davidson(A,v0=None,tol=1e-10,maxiter=1000,eigen_solver='eigh'):
    '''
    The Davidson's algorithm.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.
        :eigen_solver: str, the solver for diagonalization of G.

            * 'eigh', standard.

    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''
    N=A.shape[0]
    DA_diag=A.diagonal()
    if v0 is None:
        v0=random.random((N,1))
    v0=_normalize(v0)
    vA=A.dot(v0)
    VA=vA
    V=v0
    #initialise projected matrix.
    G=v0.T.conj().dot(vA)
    for i in xrange(maxiter):
        ei,vi=_eigen_solve(G,method=eigen_solver)
        #compute largest Ritz value theta, and Ritz vector u.
        imax=argmax(ei)
        theta,u=ei[imax],V.dot(vi[:,imax:imax+1])
        #get the residual
        r=VA.dot(vi[:,imax:imax+1])-theta*u
        if norm(r)<tol:
            return theta,u
        print '%s ||r|| = %s, e = %s'%(i,norm(r),theta)
        #compute the correction vector z
        z=-1./(DA_diag-theta)[:,newaxis]*r
        z=mgs(V,z)
        z=_normalize(z)

        vA=A.dot(z)
        #add z to search space.
        VA=concatenate([VA,vA],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(vA)],[vA.T.conj().dot(V),vA.T.conj().dot(z)]]
        G=bmat([[G,V.T.conj().dot(vA)],[vA.T.conj().dot(V),vA.T.conj().dot(z)]])
        V=concatenate([V,z],axis=1)
    return theta,u

def JD(A,v0=None,tol=1e-10,maxiter=1000,linear_solver='cgs',eigen_solver='eigh',jmax=20,jmin=5,sigma=0,converge_bound=1e-2):
    '''
    The Jacobi-Davidson's algorithm.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.
        :linear_solver: str, method to iterative solve Ax = b.
            
            * 'cgs', conjugate gradient square iteration.
            * 'gmres', Generalized Minimal RESidual iteration.
        :eigen_solver: str, the solver for diagonalization of G.

            * 'eigh', standard.
        :sigma: float, the desired eigenvalue region.
        :converge_bound: float, if current tol<converge_bound, use theta as the shift.
    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''
    N=A.shape[0]
    if v0 is None:
        v0=random.random((N,1))
    v0=_normalize(v0)
    gamma=2   #the scaling factor for correction solver

    #initialise search space and Rayleigh-Ritz procedure.
    vA=A.dot(v0)
    VA=vA
    V,u=v0,v0
    G=v0.T.conj().dot(vA)
    theta=G[0,0]
    r=vA-theta*u
    cur_tol=1.
    for i in xrange(maxiter):
        #expanding the search space.
        #correction equation: solve approximately for z
            #(I-u*u.H)(A-theta*I)(I-u*u.H)z = -r, with z.T*u = 0
        z=_jdeq_solve(A,V,r,shift=theta if cur_tol<converge_bound else sigma,method=linear_solver,K=None,tol=min(0.2,cur_tol))
        z=mgs(V,z[:,newaxis])
        z=_normalize(z)
        
        #add z to search space and compute Ritz pair
        vA=A.dot(z)
        #add z to search space.
        VA=concatenate([VA,vA],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(vA)],[vA.T.conj().dot(V),vA.T.conj().dot(z)]]
        G=bmat([[G,V.T.conj().dot(vA)],[vA.T.conj().dot(V),vA.T.conj().dot(z)]])
        V=concatenate([V,z],axis=1)
        ei,vi=_eigen_solve(G,method=eigen_solver)
        #compute largest Ritz value theta, and Ritz vector u.
        imax=argmax(ei)
        theta,u=ei[imax],V.dot(vi[:,imax:imax+1])
        #get the residual
        r=VA.dot(vi[:,imax:imax+1])-theta*u
        cur_tol=norm(r)
        if cur_tol<tol:
            return theta,u
        print '%s ||r|| = %s, e = %s'%(i,cur_tol,theta)

        #restart
        #if len(ei)>=jmax:
        #    V=

    return theta,u

