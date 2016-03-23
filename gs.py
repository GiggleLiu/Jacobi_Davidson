'''
Gram Schmidt Orthogonalization.
'''

from numpy import *
import pdb,time,warnings

__all__=['gs','mgs','icgs']

def mgs(Q,u,M=None):
    '''
    Modified Gram-Schmidt orthogonalisation,
    
    Parameters:
        :Q: matrix, the search space.
        :u: vector, the vector to be orthogonalized.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    uH=u.T.conj()
    MQ=M.dot(Q) if M is not None else Q
    for i in xrange(Q.shape[1]):
        s=uH.dot(MQ[:,i:i+1])
        u=u-s.conj()*Q[:,i:i+1]
    return u

def icgs(Q,u,M=None):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :Q: matrix, the search space.
        :u: vector, the vector to be orthogonalized.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    uH,QH=u.T.conj(),Q.T.conj()
    alpha=0.5
    itmax=3
    it=1
    Mu=M.dot(u) if M is not None else u
    r_pre=sqrt(uH.dot(Mu))
    for it in xrange(itmax):
        u=u-Q.dot(QH.dot(Mu))
        Mu=M.dot(u) if M is not None else u
        r1=sqrt(uH.dot(Mu))
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return u

def gs(Q,u,M=None):
    '''
    Classical Gram-Schmidt orthogonalisation.
    
    Parameters:
        :Q: matrix, the search space.
        :u: vector, the vector to be orthogonalized.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    u=u-Q.dot(Q.T.conj().dot(u))#((u.T.dot(Q.conj()))*Q).sum(axis=1)[:,newaxis]
    return u


