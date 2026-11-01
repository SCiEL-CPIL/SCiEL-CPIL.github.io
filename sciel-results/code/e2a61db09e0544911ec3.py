"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_decomp():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_2x3_3x2_rank11():
    Us, Vs, Ws = _strassen_2x2_decomp()
    U_s = np.zeros((6, 7)); V_s = np.zeros((6, 7)); W_s = np.zeros((4, 7))
    for i in range(2):
        for j in range(2):
            U_s[i*3+j, :] = Us[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V_s[j*2+k, :] = Vs[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W_s[k*2+i, :] = Ws[k*2+i, :]
    U_o = np.zeros((6, 4)); V_o = np.zeros((6, 4)); W_o = np.zeros((4, 4))
    for i in range(2):
        for k in range(2):
            t = i*2+k
            U_o[i*3+2, t] = 1; V_o[4+k, t] = 1; W_o[k*2+i, t] = 1
    return np.hstack([U_s, U_o]), np.hstack([V_s, V_o]), np.hstack([W_s, W_o])

def _build_full_rank44():
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _build_2x3_3x2_rank11()
    r = 11; nb = 4; R = r * nb
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    for t in range(nb):
        U[:, t*r:(t+1)*r] = U_sub
        for j in range(m):
            for kp in range(2):
                V[j*p + t*2+kp, t*r:(t+1)*r] = V_sub[j*2+kp]
        for kp in range(2):
            for i in range(n):
                W[(t*2+kp)*n+i, t*r:(t+1)*r] = W_sub[kp*n+i]
    return U, V, W

def _verify(U, V, W, n=2, m=3, p=8, tol=1e-10):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    Td = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(Td, T, atol=tol, rtol=0)

def _schoolbook(n=2, m=3, p=8):
    R = n*m*p
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    U44, V44, W44 = _build_full_rank44()
    assert _verify(U44, V44, W44), "Rank-44 verify failed"
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
# EVOLVE-BLOCK-END
