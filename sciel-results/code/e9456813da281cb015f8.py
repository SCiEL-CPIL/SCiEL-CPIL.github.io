"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140  # safety margin


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _strassen_block():
    """
    Returns 7 terms for 2x2x2 multiplication.
    U: 4 entries (A00, A01, A10, A11)
    V: 4 entries (B00, B01, B10, B11)
    W: 4 entries (C00, C01, C10, C11)
    """
    terms = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])), # P0=(A00+A11)(B00+B11) -> C00+=1, C11+=1? No, check Strassen
        # Standard Strassen:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        # C00 = P1 - P2 - P4 + P7
        # C01 = P4 + P5
        # C10 = P2 + P6
        # C11 = P1 - P3 + P4 - P6
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, -1, 0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, -1, 1, 0]), np.array([1, 0, -1, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 1, 0, 1])),
        (np.array([1, 0, -1, 0]), np.array([1, 1, 0, 0]), np.array([-1, 0, 1, 0])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])),
    ]
    # Verify P1: U=[1,0,0,1], V=[1,0,0,1]. Product: A00B00 + A00B11 + A11B00 + A11B11.
    # W=[1,0,0,1] means C00+=1, C11+=1.
    # A00B00 is in C00. A11B11 is in C11.
    # A00B11 is NOT in C00 or C11.
    # A11B00 is NOT in C00 or C11.
    # So this term alone is invalid. It relies on cancellation.
    # The W vector should be [1, 0, 0, 1] ONLY IF the cross terms cancel out.
    # In Strassen, they do.
    return terms


def _schoolbook_221():
    """
    Returns 4 terms for 2x2x1 multiplication.
    U: 4 entries
    V: 2 entries (B00, B10)
    W: 2 entries (C00, C10)
    """
    terms = [
        (np.array([1, 0, 0, 0]), np.array([1, 0]), np.array([1, 0])),
        (np.array([0, 1, 0, 0]), np.array([0, 1]), np.array([1, 0])),
        (np.array([0, 0, 1, 0]), np.array([1, 0]), np.array([0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, 1]), np.array([0, 1])),
    ]
    return terms


def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # U: 4xRank, V: 18xRank, W: 18xRank
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen_terms = _strassen_block()
    schoolbook_terms = _schoolbook_221()
    
    # 4 blocks of 2x2x2 (cols 0-1, 2-3, 4-5, 6-7)
    for block_idx in range(4):
        col_start = block_idx * 2
        # V rows for this block: B[0, col_start], B[0, col_start+1], B[1, col_start], B[1, col_start+1]
        # Indices in V (size 18):
        # V row index = j * p + k
        # j=0, k=col_start -> 0*9 + col_start = col_start
        # j=0, k=col_start+1 -> col_start+1
        # j=1, k=col_start -> 9 + col_start
        # j=1, k=col_start+1 -> 9 + col_start+1
        
        # W rows for this block: C[0, col_start], C[0, col_start+1], C[1, col_start], C[1, col_start+1]
        # Indices in W (size 18):
        # W row index = k * n + i
        # k=col_start, i=0 -> col_start * 2 + 0 = 2*col_start
        # k=col_start+1, i=0 -> (col_start+1)*2 = 2*col_start + 2
        # k=col_start, i=1 -> 2*col_start + 1
        # k=col_start+1, i=1 -> 2*col_start + 3
        
        v_indices = [col_start, col_start + 1, 9 + col_start, 9 + col_start + 1]
        w_indices = [2 * col_start, 2 * col_start + 2, 2 * col_start + 1, 2 * col_start + 3]
        
        for u, v, w in strassen_terms:
            u_full = u
            v_full = np.zeros(m * p)
            for i, idx in enumerate(v_indices):
                v_full[idx] = v[i]
            w_full = np.zeros(p * n)
            for i, idx in enumerate(w_indices):
                w_full[idx] = w[i]
            
            U_terms.append(u_full)
            V_terms.append(v_full)
            W_terms.append(w_full)
            
    # 1 block of 2x2x1 (col 8)
    col_start = 8
    v_indices = [col_start, 9 + col_start]
    w_indices = [2 * col_start, 2 * col_start + 1]
    
    for u, v, w in schoolbook_terms:
        u_full = u
        v_full = np.zeros(m * p)
        for i, idx in enumerate(v_indices):
            v_full[idx] = v[i]
        w_full = np.zeros(p * n)
        for i, idx in enumerate(w_indices):
            w_full[idx] = w[i]
            
        U_terms.append(u_full)
        V_terms.append(v_full)
        W_terms.append(w_full)
        
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    if _validate(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook if validation fails (should not happen)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
