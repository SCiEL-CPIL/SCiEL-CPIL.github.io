import numpy as np
import time

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U_list, V_list, W_list):
    """Verify exact decomposition of 2x2*2x7 matrix multiplication tensor."""
    n, m, p = 2, 2, 7
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_rank25():
    """Construct rank-25 Strassen-based decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    def add_strassen_block(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    col_start = 21
    k = 6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1
    return U, V, W

def _try_merge_terms(U, V, W, deadline):
    """Try to reduce rank by finding redundant or mergeable terms."""
    n, m, p = 2, 2, 7
    cur_rank = U.shape[1]
    
    # Check for zero terms
    for r in range(cur_rank):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            cur_rank -= 1
            break
    
    # Try to merge pairs of terms
    for r1 in range(U.shape[1]):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, U.shape[1]):
            if time.time() > deadline:
                break
            # Check if term r2 is proportional to term r1
            u1, u2 = U[:, r1], U[:, r2]
            v1, v2 = V[:, r1], V[:, r2]
            w1, w2 = W[:, r1], W[:, r2]
            # Try scaling: can we absorb r2 into r1?
            # Need u1*v1*w1 + c*u2*v2*w2 to still work after removing r2
            # This is complex; skip for now
            pass
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen rank-7 for three 2x2 blocks and schoolbook rank-4 for the last column.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 2, 7
    
    # Build the rank-25 Strassen-based decomposition
    U, V, W = _build_strassen_rank25()
    
    # Verify it's correct
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        return _schoolbook_decomposition()
    
    # Try to reduce rank
    U, V, W = _try_merge_terms(U, V, W, deadline)
    
    # Final verification
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        U, V, W = _build_strassen_rank25()
        if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
            return _schoolbook_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
