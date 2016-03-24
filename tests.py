from numpy import *
from scipy.linalg import norm
from scipy import sparse as sps
from scipy.sparse.linalg import eigsh
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time,warnings

from pydavidson import *
from pydavidson import _normalize

class Test(object):
    def __init__(self):
        N=1000
        N1=10
        mat=sps.coo_matrix((random.random(N1*N)+1j*random.random(N1*N),(random.randint(0,N,N1*N),random.randint(0,N,N1*N))),shape=(N,N))
        mat=mat.T.conj()+mat+sps.diags(random.random(N),0)*10
        self.mat=mat

    def test_davidson(self):
        mat=self.mat
        N=mat.shape[0]
        t0=time.time()
        e,v=eigsh(mat,k=1)
        t1=time.time()
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=davidson(mat,v0=None,tol=1e-10,maxiter=1000)
        t2=time.time()
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,norm(v.T.conj().dot(v2)),t2-t1,t1-t0)
        pdb.set_trace()

    def test_jocc(self):
        mat=self.mat
        mat[0,0]=10
        t0=time.time()
        e,v=JOCC(mat)
        v=_normalize(v)
        t1=time.time()
        e2,v2=eigsh(mat,k=1)
        t2=time.time()
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e,e2,norm(v.T.conj().dot(v2)),t1-t0,t2-t1)
        pdb.set_trace()

    def test_jd(self):
        mat=self.mat
        N=mat.shape[0]
        t0=time.time()
        e,v=eigsh(mat,k=2,which='LA')
        t1=time.time()
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=JDh(mat,k=2,v0=None,tol=1e-10,maxiter=1000,which='LA',sigma=20,linear_solver_maxiter=20,linear_solver='bicgstab')
        t2=time.time()
        print 'E = %s(Exact = %s), fidelity -> %s, Elapse -> %s(%s).'%(e2,e,norm(v.T.conj().dot(v2)),t2-t1,t1-t0)
        pdb.set_trace()

    def test_all(self):
        self.test_jd()
        self.test_davidson()
        self.test_jocc()

Test().test_all()
