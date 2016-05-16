'''
Gram Schmidt Orthogonalization.
'''

from numpy import *
import pdb,time,warnings

__all__=['gs','mgs','icgs']

def mgs(u,Q,MQ=None,M=None):
    '''
    Modified Gram-Schmidt orthogonalisation,
    
    Parameters:
        :u: vector, the vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :MQ: matrix, the matrix of M*Q, if provided, perform M-orthogonal.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    uH=u.T.conj()
    if MQ is None:
        MQ=M.dot(Q) if M is not None else Q
    for i in xrange(Q.shape[1]):
        s=uH.dot(MQ[:,i:i+1])
        u=u-s.conj()*Q[:,i:i+1]
    return u

def icgs(u,Q,M=None,colwise=True,return_norm=False):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :colwise: bool, column wise orthogonalization.
        :return_norm: bool, return the norm of u.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    assert(M is None or colwise)
    uH,QH=u.T.conj(),Q.T.conj()
    alpha=0.5
    itmax=3
    it=1
    Mu=M.dot(u) if M is not None else u
    r_pre=sqrt(abs(uH.dot(Mu))) if colwise else sqrt(abs(Mu.dot(uH)))
    for it in xrange(itmax):
        if colwise:
            u=u-Q.dot(QH.dot(Mu))
            Mu=M.dot(u) if M is not None else u
            r1=sqrt(abs(uH.dot(Mu)))
        else:
            u=u-u.dot(QH).dot(Q)
            r1=sqrt(abs(u.dot(uH)))
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u

def gs(u,Q,M=None,colwise=True):
    '''
    Classical Gram-Schmidt orthogonalisation.
    
    Parameters:
        :u: vector, the vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :colwise: bool, column wise orthogonalization.

    Return:
        vector, orthogonalized vector u.
        Note: the resulting vetors are not normalized! and in the confusion cases, columnwise orthogonalization is prefered.
    '''
    assert(ndim(u)==2)
    assert(M is None or colwise)
    if colwise:
        u=u-Q.dot(Q.T.conj().dot(u))
    else:
        u=u-u.dot(Q.T.conj()).dot(Q)
    return u


