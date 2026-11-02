"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for c_idx in range(rank):
                    val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                if abs(val - 1.0) > tol:
                    return False
    # Check some non-diagonal entries
    for _ in range(100):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j != j2 or k != k2:
            val = 0.0
            for c_idx in range(rank):
                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
            if abs(val) > tol:
                return False
    return True

def _get_strassen_terms():
    """
    Returns U, V, W coefficients for 2x2x2 Strassen decomposition.
    Local indices i', j', k' in {0, 1}.
    U is 4x7, V is 4x7, W is 4x7.
    """
    # 7 terms
    # U[i'*2 + j', r]
    # V[j'*2 + k', r]
    # W[k'*2 + i', r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # u: 00=1, 11=1
    U[0, 0] = 1; U[3, 0] = 1
    # v: 00=1, 11=1
    V[0, 0] = 1; V[3, 0] = 1
    # w: 00=1, 11=1, 01=0, 10=0
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # u: 10=1, 11=1
    U[2, 1] = 1; U[3, 1] = 1
    # v: 00=1
    V[0, 1] = 1
    # w: 10=1, 11=-1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    # u: 00=1
    U[0, 2] = 1
    # v: 10=1, 11=-1
    V[2, 2] = 1; V[3, 2] = -1
    # w: 01=1, 11=1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    # u: 11=1
    U[3, 3] = 1
    # v: 01=1, 00=-1
    V[1, 3] = 1; V[0, 3] = -1
    # w: 00=1, 10=1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 4: M5 = (A00+A10)B11
    # u: 00=1, 10=1
    U[0, 4] = 1; U[2, 4] = 1
    # v: 11=1
    V[3, 4] = 1
    # w: 00=-1, 01=1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # u: 10=1, 00=-1
    U[2, 5] = 1; U[0, 5] = -1
    # v: 00=1, 01=1
    V[0, 5] = 1; V[1, 5] = 1
    # w: 11=1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # u: 01=1, 11=-1
    U[1, 6] = 1; U[3, 6] = -1
    # v: 10=1, 11=1
    V[2, 6] = 1; V[3, 6] = 1
    # w: 00=1
    W[0, 6] = 1
    
    return U, V, W

def _add_strassen_block(U, V, W, rank_offset, i_start, j_start, k_start, n, m, p):
    """
    Adds a Strassen block for indices:
    i in [i_start, i_start+1]
    j in [j_start, j_start+1]
    k in [k_start, k_start+1]
    """
    U_loc, V_loc, W_loc = _get_strassen_terms()
    
    for r in range(7):
        col = rank_offset + r
        for i_loc in range(2):
            for j_loc in range(2):
                i_glob = i_start + i_loc
                j_glob = j_start + j_loc
                if U_loc[i_loc*2 + j_loc, r] != 0:
                    U[i_glob * m + j_glob, col] += U_loc[i_loc*2 + j_loc, r]
        
        for j_loc in range(2):
            for k_loc in range(2):
                j_glob = j_start + j_loc
                k_glob = k_start + k_loc
                if V_loc[j_loc*2 + k_loc, r] != 0:
                    V[j_glob * p + k_glob, col] += V_loc[j_loc*2 + k_loc, r]
        
        for k_loc in range(2):
            for i_loc in range(2):
                k_glob = k_start + k_loc
                i_glob = i_start + i_loc
                if W_loc[k_loc*2 + i_loc, r] != 0:
                    W[k_glob * n + i_glob, col] += W_loc[k_loc*2 + i_loc, r]

def _add_schoolbook_term(U, V, W, rank_offset, i, j, k, n, m, p):
    col = rank_offset
    U[i * m + j, col] = 1
    V[j * p + k, col] = 1
    W[k * n + i, col] = 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use rank-1 terms for all (i,j,k) triples: rank = n*m*p = 45
    target_rank = n * m * p
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    rank_offset = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                _add_schoolbook_term(U, V, W, rank_offset, i, j, k, n, m, p)
                rank_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
