'''
Jacobi-Davidson eigenvalue solver for hermitian(complex symmetric) matrices.

Author: JinGuo Leo
Year: 2016

Reference:
    Geus, R. (2002). The Jacobi-Davidson algorithm for solving large sparse symmetric eigenvalue problems with application to the design of accelerator cavities, (14734). 
    Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.9572&rep=rep1&type=pdf

Dependency:
    numpy, scipy(latest version)
'''

from numpy import *
from scipy import sparse as sps
from scipy.linalg import eigh,inv
from scipy.linalg.lapack import dsyev
from numpy.linalg import norm
from scipy.sparse import linalg as lin
from scipy.sparse.linalg import inv as spinv
import pdb,time,warnings

from gs import *

__all__=['davidson_Basic','JOCC_Basic','JDh']

def _normalize(v0):
    '''Calculate the norm.'''
    if ndim(v0)==2:
        return v0/sqrt((multiply(v0,v0.conj())).sum(axis=0))
    elif ndim(v0)==1:
        return v0/norm(v0)

class _JDLinear(lin.LinearOperator):
    '''
    Jacobi-Davidson Linear Operator.

    Attributes:
        :lprojector/rprojector: tuple, (QL,QR) to form (I-QL*QR)
        :spmat: matrix, the sparse matrix on the right hand side.
    '''
    def __init__(self,lprojector,spmat,rprojector=None):
        self.lprojector=lprojector
        self.rprojector=rprojector
        self.spmat=spmat
        spshape=spmat.shape if not isinstance(spmat,tuple) else spmat[0].shape
        super(_JDLinear,self).__init__(shape=spshape,dtype=spmat[0].dtype if isinstance(spmat,tuple) else spmat.dtype)

    @staticmethod
    def _apply_projector(x,projector,right=False):
        '''apply projector 1-projector[0]*projector[1]. to x'''
        if projector is None:return x
        if right:
            res=x-x.dot(projector[0]).dot(projector[1])
        else:
            res=x-projector[0].dot(projector[1].dot(x))
        return res

    @staticmethod
    def _apply_center(x,mats,right=False):
        '''apply center matrices one by one.'''
        if mats is None:
            return x
        res=x
        if not isinstance(mats,tuple):
            mats=(mats,)
        if right:
            for mat in mats:
                res=res*mat
        else:
            for mat in mats[::-1]:
                res=mat*res
        return res

    def _matvec(self,x):
        '''Matrix vector multiplication.'''
        res=self._apply_projector(x,self.rprojector)
        res=self._apply_center(res,self.spmat)
        res=self._apply_projector(res,self.lprojector)
        return res

    def _rmatvec(self,x):
        res=self._apply_projector(x,self.lprojector,right=True)
        res=self._apply_center(res,self.spmat,right=True)
        res=self._apply_projector(res,self.rprojector,right=True)
        return res

def _jdeq_solve(A,Q,MQ,iKMQ,r,M,invK,F,shift,tol,method='lgmres',precon=False,maxiter=20):
    '''
    Solve Jacobi-Davidson's correction equation.
        (I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*Q = 0

    Parameters:
        :A: matrix(N,N), the matrix.
        :Q,MQ,iKMQ: matrix(N,k), Q,M*Q and K^-1*M*Q, Q is the search space, z is orthogonal to Q.
        :r: 1D array, the distance.
        :M: matrix(N,N), the preconditioner for JD.
        :invK: matrix(N,N), part of preconditioner for this solver.
        :F: matrix(k,k), the space of converged eigenvectors.
        :shift: float, the disired eigenvalue region.
        :tol: float, the tolerence.
        :method: str, the method to solve linear equation.

            * 'gmres'/'lgmres', generalized Mininal residual iteration.
            * 'cg'/'bicg'/'bicgstab', (Bi)conjugate gradient method.
            * not used methods, `minres(minimum residual)` and `qmr(quasi-minimum residual)` are designed for real symmetric matrices,\
                    `cgs` and `symmlq` is not used for the same reason.\
                    `lsqr`/`lsmr`(least square methods) are working poorly here.
        :precon: bool, use preconditioner if True.

    Return:
        1D array, the result z.
    '''
    N,k=MQ.shape
    MQH=MQ.T.conj()
    MAT=A-shift*(M if M is not None else sps.identity(N))
    invF=inv(F)
    if not precon:
        if invK is not None:
            MAT=(invK,MAT)
            r=invK.dot(r)
        system_matrix=_JDLinear((iKMQ.dot(invF),MQH),MAT)
        precon=None
        right_hand_side=-r+iKMQ.dot(invF).dot(MQH.dot(r))
    else:
        QH=Q.T.conj()
        system_matrix=_JDLinear((MQ,QH),MAT)
        precon=_JDLinear((iKMQ.dot(invF),MQH),invK if invK is not None else sps.identity(N))
        right_hand_side=-r+Q.dot(QH.dot(r))
    if method=='cg' or method=='bicg' or method=='bicgstab':
        if method=='cg': solver=lin.cg 
        elif method=='bicg': solver=lin.bicg
        else: solver=lin.bicgstab
        z,info=solver(system_matrix,right_hand_side,tol=tol,M=precon,maxiter=maxiter)
    elif method=='gmres' or method=='lgmres':
        if method=='gmres': solver=lin.gmres
        else: solver=lin.lgmres
        z,info=solver(system_matrix,right_hand_side,tol=tol,M=precon,maxiter=maxiter)
    else:
        raise Exception('Unknown method for linear solver %s'%method)
    return z

def JOCC_Basic(A,maxiter=1000,tol=1e-15):
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

def davidson_Basic(A,v0=None,tol=1e-10,maxiter=1000):
    '''
    The Davidson's algorithm.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.

    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''
    N=A.shape[0]
    A=A.tocsr()
    DA_diag=A.diagonal()
    if v0 is None:
        v0=random.random((N,1))
    elif ndim(v0)==1: 
        v0=v0[:,newaxis]
    v0=_normalize(v0)
    Av=A.dot(v0)
    AV=Av
    V=v0
    #initialise projected matrix.
    G=v0.T.conj().dot(Av)
    for i in xrange(maxiter):
        ei,vi=eigh(G)
        #compute largest Ritz value theta, and Ritz vector u.
        imax=argmax(ei)
        theta,u=ei[imax],V.dot(vi[:,imax:imax+1])
        #get the residual
        r=AV.dot(vi[:,imax:imax+1])-theta*u
        if norm(r)<tol:
            return theta,u
        print '%s ||r|| = %s, e = %s'%(i,norm(r),theta)
        #compute the correction vector z
        z=-1./(DA_diag-theta)[:,newaxis]*r
        z=mgs(z,V)
        z=_normalize(z)

        Av=A.dot(z)
        #add z to search space.
        AV=concatenate([AV,Av],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]]
        G=bmat([[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]])
        V=concatenate([V,z],axis=1)
    return theta,u

def JDh(A,v0=None,k=1,which='SL',M=None,K=None,tol=1e-10,maxiter=1000,projector=None,\
        linear_solver='bicgstab',linear_solver_maxiter=20,linear_solver_precon=False,\
        jmax=20,jmin=5,sigma=None,converge_bound=1e-3,gap_estimate=0.1,iprint=0):
    '''
    The Jacobi-Davidson's algorithm for the Hermitian matrix.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :k: int, the number of eigenvalues.
        :which: str,

            * 'SA', the one with smallest eigenvalue.
            * 'LA', the one with largest eigenvalue.
            * 'SL', the one cloest to sigma.
        :M: matrix/None, the preconditioner.
        :K: matrix/None, the "core" of the preconditioner to iterative solve J-D equation, it approximates A-sigma*M and cheap to invert.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.
        :projector: matrix/LinearOperator, the matrix to projector the vector to desired space, often used to avoid degeneracy(commute with A is desired).
        :linear_solver: str, method to iterative solve Ax = b.
            
            * 'gmres'/'lgmres', generalized Mininal residual iteration.
            * 'cg'/'bicg'/'bicgstab', (Bi)conjugate gradient method.
        :linear_solver_maxiter: int, the maximum iteration for linear solver.
        :linear_solver_precon: bool, use precondioner in linear solver if True.
        :jmax/jmin: int, the maximum and minimum working space.
        :sigma: float/None, the desired eigenvalue region, None for not using sigma(if which=='SL' is used, use initial v0 as the critetia).
        :converge_bound: float, if current tol<converge_bound, use theta as the shift.
        :gap_estimate: float, for which = LA/SA, the gap_estimate is the distance between sigma and the smallest/largest eigen values.
        :iprint: int, the amount of details to be printed, 0 for not print, 10 for debug mode.

    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''
    if sigma is None and v0 is None and which=='SL':
        raise ValueError('You must specify a desired energy or initial vector in \'SL\' mode!')
    N=A.shape[0]
    A=A.tocsr()
    if M is not None: M=M.tocsr()
    if projector is not None: projector=projector.tocsr()
    if v0 is None:
        v0=random.random((N,1))-0.5
    elif ndim(v0)==1: 
        v0=v0[:,newaxis]
    if projector is not None:
        v0=projector.dot(v0)
    v0=_normalize(v0)
    if K is not None:
        t0=time.time()
        luobj=lin.splu(K.tocsc(),permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True,'ILU_MILU':'SILU','ILU_DropTol':1e-4,'ILU_FillTol':1e-2})
        invK=lin.LinearOperator((N,N),matvec=luobj.solve,matmat=luobj.solve,dtype=A.dtype)
        t1=time.time()
        if iprint>0:
            print 'Time used for calculate preconditioner K ->',t1-t0
    else:
        invK=None

    #initialise search space and Rayleigh-Ritz procedure.
    Av=A.dot(v0)
    AV=Av
    V,u=v0,v0
    G=v0.T.conj().dot(Av)
    theta=G[0,0]
    if which=='SL' and sigma is None: sigma=theta  #initialize sigma by v0!
    conv_steps,lambs,Q,F=[0],[],zeros([N,0]),zeros([0,0])  #the eigenvalues and eigenvectors.
    MQ,iKMQ=Q,Q  #M*Q and K^-1*M*Q

    for i in xrange(maxiter):
        #solve G and compute largest Ritz value theta, and Ritz vector u.
        S,W=eigh(G)  #the spectrum S is in ascending order.
        if which=='SL':
            distance=abs(S-sigma)
            order=argsort(distance)
            S,W=S[order],W[:,order]
        elif which=='LA':
            S,W=S[::-1],W[:,::-1]

        #store converged pairs
        while True:
            #Ritz approximation.
            theta,u=S[0],V.dot(W[:,:1])
            Mu=M.dot(u) if M is not None else u
            iKMu=invK.dot(Mu) if invK is not None else Mu

            #get the residual
            r=A.dot(u)-theta*Mu
            cur_tol=norm(r)
            if which=='LA': sigma=theta+gap_estimate
            elif which=='SA': sigma=theta-gap_estimate
            if iprint>0:
                print '%s ||r|| = %s, e = %s'%(i,cur_tol,theta)

            Q_=concatenate([Q,u],axis=1)
            MQ_=concatenate([MQ,Mu],axis=1)
            iKMQ_=concatenate([iKMQ,iKMu],axis=1)
            F_=bmat([[F,MQ.T.conj().dot(iKMu)],[iKMu.T.conj().dot(MQ),Mu.T.conj().dot(iKMu)]])
            if cur_tol>tol or (len(lambs)!=k-1 and len(S)<=1):
                #continue the outer run if
                #1. meet an unconverged pair.
                #2. or, we are going to get the non-last eigenvalue, but will empty S, which will raise Error in the next run.
                break
            #move a converged pair from V to lambs and Q
            lambs.append(theta);conv_steps.append(i)
            if iprint>0:
                print 'Find eigenvalue %s'%theta
            Q,MQ,iKMQ,F=Q_,MQ_,iKMQ_,F_
            V,S=V.dot(W[:,1:]),S[1:]
            G,W=diag(S),identity(S.shape[0])
            if len(lambs)==k:
                return array(lambs),Q

        #restart
        if len(S)==jmax:
            V,S=V.dot(W[:,:jmin]),S[:jmin]
            G,W=diag(S),identity(S.shape[0])

        #compute the shift
        shift=theta if cur_tol<converge_bound else sigma

        #correction equation: solve approximately for z:
        #     (I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*u = 0
        ctol=cur_tol*3e-2
        z=_jdeq_solve(A,r=r,Q=Q_,MQ=MQ_,iKMQ=iKMQ_,invK=invK,M=M,F=F_,shift=shift,\
                method=linear_solver,tol=ctol,precon=linear_solver_precon,maxiter=linear_solver_maxiter)[:,newaxis]
        
        if iprint==10:
            Pz=z-Q_.dot(Q_.T.conj().dot(z))
            C=A-shift*M
            print 'JD equation tol',norm(P.dot(C.dot(Pz))+r)
            pdb.set_trace()

        #project to correct subspace.
        if projector is not None:
            z=projector.dot(z)
        z=mgs(z,Q_,MQ=MQ_)
        z,z_norm=icgs(z,V,M=M,return_norm=True)  #orthogonal to search space V.
        z=z/z_norm
        if iprint==10 and projector is not None:
            if not abs(z.T.conj().dot(projector.dot(z)))>0.99:
                pdb.set_trace()
        
        #add z to search space and compute Ritz pair
        Av=A.dot(z)
        #add z to search space.
        AV=concatenate([AV,Av],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]]
        G=bmat([[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]])
        V=concatenate([V,z],axis=1)
    return array(lambs),Q

