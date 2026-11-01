"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_tensor_flat(n, m, p):
    """Build the multiplication tensor as a list of (U_idx, V_idx, W_idx, value) tuples."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx, 1.0))
    return entries

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    entries = _build_tensor_flat(n, m, p)
    for u_idx, v_idx, w_idx, val in entries:
        computed = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        if abs(computed - val) > tol:
            return False
    return True

def _strassen_block():
    """Return (U, V, W) for 2x2x2 Strassen."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[1,5]=1
    U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[3,6]=1
    return U, V, W

def _schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _construct_rank53():
    n, m, p = 2, 2, 15
    U = np.zeros((4, 53))
    V = np.zeros((30, 53))
    W = np.zeros((30, 53))
    sU, sV, sW = _strassen_block()
    for b in range(7):
        k_off = 2*b; r_off = 7*b
        U[:, r_off:r_off+7] += sU
        for j in range(2):
            for k in range(2):
                V[j*15+k_off+k, r_off:r_off+7] += sV[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W[(k_off+k)*2+i, r_off:r_off+7] += sW[k*2+i, :]
    hU, hV, hW = _schoolbook_2x2x1()
    r_off = 49; k_off = 14
    U[:, r_off:r_off+4] += hU
    for j in range(2):
        V[j*15+k_off, r_off:r_off+4] += hV[j, :]
    for i in range(2):
        W[k_off*2+i, r_off:r_off+4] += hW[i, :]
    return U, V, W

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 130
    
    U, V, W = _construct_rank53()
    best_rank = 53
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find rank-52 by checking if any rank dimension can be eliminated
    # via linear combination with others
    # This is a heuristic search
    
    # For now, return the rank-53 solution
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
# EVOLVE-BLOCK-END
