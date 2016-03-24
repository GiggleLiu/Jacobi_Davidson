'''
Davidson and Jacobi-Davidson eigenvalue solver.
'''

from numpy import *
from scipy import sparse as sps
from scipy.linalg import eigh,inv
from scipy.linalg.lapack import dsyev
from numpy.linalg import norm
from scipy.sparse import linalg as lin
from scipy.sparse.linalg import inv as spinv
from pykrylov.symmlq import Symmlq
import pdb,time,warnings

from gs import *

__all__=['davidson','JOCC','JDh']

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

class JDLinear(lin.LinearOperator):
    '''
    Jacobi-Davidson Linear Operator.

    Attributes:
        :lprojector/rprojector: tuple, (QL,QR) to form (I-QL*QR)
        :spmat: matrix, the sparse matrix on the right hand side.
    '''
    elapse=0
    def __init__(self,lprojector,spmat,rprojector=None):
        self.lprojector=lprojector
        self.rprojector=rprojector
        self.spmat=spmat
        spshape=spmat.shape if not isinstance(spmat,tuple) else spmat[0].shape
        super(JDLinear,self).__init__(spshape,matvec=self._matvec,rmatvec=self._rmatvec,dtype=complex128)

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
        t0=time.time()
        res=self._apply_projector(x,self.rprojector)
        res=self._apply_center(res,self.spmat)
        res=self._apply_projector(res,self.lprojector)
        t1=time.time()
        self.elapse+=t1-t0
        return res

    def _rmatvec(self,x):
        t0=time.time()
        res=self._apply_projector(x,self.lprojector,right=True)
        res=self._apply_center(res,self.spmat,right=True)
        res=self._apply_projector(res,self.rprojector,right=True)
        t1=time.time()
        self.elapse+=t1-t0
        return res

    def transpose(self):
        '''transpose this operator'''
        lprojector,rprojector=(self.rprojector[1].T,self.rprojector[0].T),(self.lprojector[1].T,self.lprojector[0].T)
        if isinstance(spmat,tuple):
            spmat=tuple(m.T for m in spmat[::-1])
        else:
            spmat=spmat.T
        return JDLinear(lprojector,spmat,rprojector)

    def adjoint(self):
        '''transpose this operator'''
        lprojector,rprojector=(self.rprojector[1].T.conj(),self.rprojector[0].T.conj()),(self.lprojector[1].T.conj(),self.lprojector[0].T.conj())
        if isinstance(spmat,tuple):
            spmat=tuple(m.conj().T for m in spmat[::-1])
        else:
            spmat=spmat.conj().T
        return JDLinear(lprojector,spmat,rprojector)

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

            * 'lsqr'/'lsmr'(working poorly!), least square, can not be used in solving symmetric matrix and can not use preconditioner.
            * 'cg'/'bicg', (Bi)conjugate gradient method.
            * `minres(minimum residual)` and `qmr(quasi-minimum residual)` are designed for real symmetric matrices,\
                    `cgs` is also for real matrices.
            * possible methods like `symmlq` are not implemented in scipy.
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
        system_matrix=JDLinear((iKMQ.dot(invF),MQH),MAT)
        precon=None
        right_hand_side=-r+iKMQ.dot(invF).dot(MQH.dot(r))
    else:
        QH=Q.T.conj()
        system_matrix=JDLinear((MQ,QH),MAT)
        precon=JDLinear((iKMQ.dot(invF),MQH),invK if invK is not None else sps.identity(N))
        right_hand_side=-r+Q.dot(QH.dot(r))
    if method=='cg' or method=='bicg' or method=='bicgstab':
        if method=='cg': solver=lin.cg 
        elif method=='bicg': solver=lin.bicg
        else: solver=lin.bicgstab
        z,info=solver(system_matrix,right_hand_side,tol=tol,M=precon,maxiter=maxiter)
    elif method=='lsqr' or method=='lsmr':
        raise Exception('Not Working properly, avoid using it!')
        if method=='lsmr': 
            z=lin.lsmr(system_matrix,right_hand_side,atol=tol,maxiter=maxiter)[0]
        else:
            z=lin.lsqr(system_matrix,right_hand_side,atol=tol,iter_lim=maxiter)[0]
    elif method=='gmres' or method=='lgmres':
        if method=='gmres': solver=lin.gmres
        else: solver=lin.lgmres
        z,info=solver(system_matrix,right_hand_side,tol=tol,M=precon,maxiter=maxiter)
    elif method=='symmlq':
        raise Exception('Not Working for complex matrices!')
        solver=Symmlq(system_matrix,precon=precon)
        solver.solve(right_hand_side[:,0],rtol=tol)
        z=solver.bestSolution
    else:
        raise Exception('Unknown method for linear solver %s'%method)
    el=system_matrix.elapse
    print el
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
    Av=A.dot(v0)
    AV=Av
    V=v0
    #initialise projected matrix.
    G=v0.T.conj().dot(Av)
    for i in xrange(maxiter):
        ei,vi=_eigen_solve(G,method=eigen_solver)
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
        linear_solver='bicgstab',linear_solver_maxiter=20,linear_solver_precon=False,eigen_solver='eigh',\
        jmax=20,jmin=5,sigma=0,converge_bound=1e-3,debug=False):
    '''
    The Jacobi-Davidson's algorithm for the Hermitian matrix.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :which: str,

            * 'SA', the one with smallest eigenvalue.
            * 'LA', the one with largest eigenvalue.
            * 'SL', the one cloest to sigma.
        :k: int, the number of eigenvalues.
        :M: matrix, the preconditioner.
        :K: matrix, the "core" of the preconditioner to iterative solve J-D equation, it approximates A-sigma*M and cheap to invert.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.
        :projector: matrix/LinearOperator, the matrix to projector the vector to desired space, often used to avoid degeneracy(commute with A is desired).
        :linear_solver: str, method to iterative solve Ax = b.
            
            * 'gmres', Generalized Minimal RESidual iteration.
        :linear_solver_maxiter: int, the maximum iteration for linear solver.
        :linear_solver_precon: bool, use precondioner in linear solver if True.
        :eigen_solver: str, the solver for diagonalization of G.

            * 'eigh', standard.
        :sigma: float, the desired eigenvalue region.
        :converge_bound: float, if current tol<converge_bound, use theta as the shift.
        :debug: bool, debug mode if True.

    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''
    sigma_shift=1e-1
    N=A.shape[0]
    if v0 is None:
        v0=random.random((N,1))
    if projector is not None:
        v0=projector.dot(v0)
    v0=_normalize(v0)
    #gamma=2   #the scaling factor for correction solver
    if K is None and which=='SL': K=A-sigma*sps.identity(N)
    if K is not None:
        luobj=splu(K.tocsc(),permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True,'ILU_MILU':'SILU'})
        invK=lin.LinearOperator((N,N),matvec=luobj.solve,matmat=luobj.solve,dtype=complex128)
    else:
        invK=None

    #initialise search space and Rayleigh-Ritz procedure.
    Av=A.dot(v0)
    AV=Av
    V,u=v0,v0
    G=v0.T.conj().dot(Av)
    theta=G[0,0]
    r=Av-theta*u
    cur_tol,shift,r0_norm=1.,sigma,norm(r)
    conv_steps,lambs,Q,F=[],[],zeros([N,0]),zeros([0,0])  #the eigenvalues and eigenvectors.
    MQ,iKMQ=Q,Q  #M*Q and K^-1*M*Q

    tt=0
    for i in xrange(maxiter):
        #solve G and compute largest Ritz value theta, and Ritz vector u.
        S,W=_eigen_solve(G,method=eigen_solver)  #the spectrum S is in ascending order.
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
            r=A.dot(u)-theta*u
            cur_tol=norm(r)
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
            #add
            lambs.append(theta);conv_steps.append(i)
            print 'Find eigenvalue %s'%theta
            Q,MQ,iKMQ,F=Q_,MQ_,iKMQ_,F_
            #remove
            V,S=V.dot(W[:,1:]),S[1:]
            G,W=diag(S),identity(S.shape[0])
            if len(lambs)==k:
                print tt
                return lambs,Q

        #restart
        if len(S)==jmax:
            V,S=V.dot(W[:,:jmin]),S[:jmin]
            G,W=diag(S),identity(S.shape[0])

        #compute the shift
        shift=theta if cur_tol<converge_bound else sigma

        #correction equation: solve approximately for z
            #(I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*u = 0
        ctol=cur_tol*1e-1#gamma**-((i-conv_steps[-1] if len(conv_steps)>0 else i)+1)
        t0=time.time()
        z=_jdeq_solve(A,r=r,Q=Q_,MQ=MQ_,iKMQ=iKMQ_,invK=invK,M=M,F=F_,shift=shift,\
                method=linear_solver,tol=ctol,precon=linear_solver_precon,maxiter=linear_solver_maxiter)[:,newaxis]
        tt+=time.time()-t0
        
        if debug:
            P=sps.identity(N)-Q_.dot(Q_.T.conj())
            C=A-sigma*sps.identity(N)
            print 'JD equation tol',norm(P.dot(C.dot(P.dot(z)))+r)
            pdb.set_trace()
        if projector is not None:
            z=projector.dot(z)
        z=mgs(z,Q_,MQ=MQ_)
        z,z_norm=icgs(z,V,return_norm=True)  #orthogonal to search space V.
        z=z/z_norm
        if debug and projector is not None:
            assert(abs(z.T.conj().dot(projector.dot(z)))>0.99)
        
        #add z to search space and compute Ritz pair
        Av=A.dot(z)
        #add z to search space.
        AV=concatenate([AV,Av],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]]
        G=bmat([[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]])
        V=concatenate([V,z],axis=1)
    return theta,u

