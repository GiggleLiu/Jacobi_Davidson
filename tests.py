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
        mat=sps.coo_matrix((random.random(5*N)+1j*random.random(5*N),(random.randint(0,N,5*N),random.randint(0,N,5*N))),shape=(N,N))
        mat=mat.T.conj()+mat
        self.mat=mat

    def test_davidson(self):
        mat=self.mat
        N=mat.shape[0]
        e,v=eigsh(mat,k=1)
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=davidson(mat,v0=None,tol=1e-10,maxiter=1000)
        print 'E = %s(Exact = %s), fidelity -> %s.'%(e,e2,norm(v.T.conj().dot(v2)))
        pdb.set_trace()

    def test_jocc(self):
        mat=self.mat
        mat[0,0]=10
        e,v=JOCC(mat)
        v=_normalize(v)
        e2,v2=eigsh(mat,k=1)
        print 'E = %s(Exact = %s), fidelity -> %s.'%(e,e2,norm(v.T.conj().dot(v2)))
        pdb.set_trace()

    def test_jd(self):
        mat=self.mat
        N=mat.shape[0]
        e,v=eigsh(mat,k=1)
        #perturb=random.random((N,1))*1e-2
        #v=_normalize(v+perturb)
        e2,v2=JD(mat,v0=None,tol=1e-10,maxiter=1000)
        print 'E = %s(Exact = %s), fidelity -> %s.'%(e,e2,norm(v.T.conj().dot(v2)))
        pdb.set_trace()

    def test_all(self):
        self.test_davidson()
        self.test_jocc()
        #self.test_jd()

Test().test_all()
