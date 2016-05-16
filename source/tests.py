from numpy import *
from scipy.linalg import norm,eigvalsh,eigh
from scipy import sparse as sps
from scipy.sparse.linalg import eigsh,splu
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time,warnings

from pydavidson import *
from pydavidson import _normalize

#to be tested list
#1. generalized eigenvalue problem with M
#2. optimize the solver for inv(A-sigma*I) with splu.
#3. projector.

class Test(object):
    def __init__(self):
        N=1000
        N1=10
        mat=sps.coo_matrix((random.random(N1*N)+1j*random.random(N1*N),(random.randint(0,N,N1*N),random.randint(0,N,N1*N))),shape=(N,N))
        mat=mat.T.conj()+mat+sps.diags(random.random(N),0)*10
        self.mat=mat
        self.M=sps.diags(random.random(N),0)

    def test_davidson(self):
        '''test for davidson basic algorithm'''
        print 'Testint for basic Davidson Algorithm ...'
        mat=self.mat
        N=mat.shape[0]
        t0=time.time()
        e,v=eigsh(mat,k=1)
        t1=time.time()
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=davidson_Basic(mat,v0=None,tol=1e-10,maxiter=1000)
        t2=time.time()
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,norm(v.T.conj().dot(v2)),t2-t1,t1-t0)
        assert_almost_equal(e,e2)
        assert_almost_equal(norm(v.T.conj().dot(v2))**2,1)

    def test_jocc(self):
        '''Test for JOCC_Basic'''
        print 'Testint for basic Jacobi Algorithm ...'
        mat=self.mat
        mat[0,0]=10
        t0=time.time()
        e,v=JOCC_Basic(mat)
        v=_normalize(v)
        t1=time.time()
        e2,v2=eigsh(mat,k=1)
        t2=time.time()
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e,e2,norm(v.T.conj().dot(v2)),t1-t0,t2-t1)
        assert_almost_equal(e,e2)
        assert_almost_equal(norm(v.T.conj().dot(v2)),1,decimal=3)

    def test_jd_min(self):
        '''get minimum eigenvalues.'''
        print 'Testint for Jacobi-Davidson Algorithm to find mimimum eigenvalues ...'
        mat=self.mat
        N=mat.shape[0]
        k=10  #10 state are calculated
        t0=time.time()
        e,v=eigsh(mat,k=k,which='SA')
        t1=time.time()
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=JDh(mat,k=k,v0=None,tol=1e-12,maxiter=1000,which='SA',sigma=5,linear_solver_maxiter=20,linear_solver='bicgstab',linear_solver_precon=True)
        t2=time.time()
        fid=norm(v.T.conj().dot(v2))**2/k
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,fid,t2-t1,t1-t0)
        assert_allclose(e,e2)
        assert_almost_equal(fid,1)

    def test_jd_sigma(self):
        '''get eigenvalues near sigma'''
        print 'Testint for Jacobi-Davidson Algorithm to find eigenvalues in specific regions ...'
        sigma=0
        mat=self.mat
        N=mat.shape[0]
        k=10
        t0=time.time()
        K=mat-sigma*sps.identity(N)
        e2,v2=JDh(mat,k=k,v0=None,K=K,tol=1e-12,maxiter=1000,which='SL',sigma=0,linear_solver_maxiter=20,linear_solver='bicgstab',linear_solver_precon=True,iprint=1)
        t1=time.time()
        e,v=eigh(mat.toarray())
        e=e[argsort(abs(e))][:k]
        t2=time.time()
        fid=norm(v.T.conj().dot(v2))**2/k
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,fid,t1-t0,t2-t1)
        assert_allclose(e,e2)
        assert_almost_equal(fid,1)

    def test_jd_general(self):
        '''Test the generalized eigenvalue problem.'''
        print 'Testint for Jacobi-Davidson Algorithm to find smallest eigenvalues for generalized eigenvalue problem with M specified! ...'
        mat=self.mat
        M=self.M
        N=mat.shape[0]
        k=1
        t0=time.time()
        e,v=eigsh(mat.tocsc(),k=k,which='SA',M=M.tocsc())
        t1=time.time()
        e2,v2=JDh(mat,k=k,v0=None,M=M,tol=1e-5,maxiter=1000,which='SA',sigma=-500,linear_solver_maxiter=20,linear_solver='lgmres',linear_solver_precon=True,iprint=0)
        t2=time.time()
        fid=norm(v.T.conj().dot(M.dot(v2)))**2/k
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,fid,t2-t1,t1-t0)
        assert_allclose(e,e2)
        assert_almost_equal(fid,1)

    def test_all(self):
        self.test_jd_min()
        self.test_davidson()
        self.test_jocc()
        self.test_jd_sigma()
        self.test_jd_general()

Test().test_all()
