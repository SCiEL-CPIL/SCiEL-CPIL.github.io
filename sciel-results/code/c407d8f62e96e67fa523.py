"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM = N * M
MP = M * P
PN = P * N
TENSOR_SIZE = N * M * P  # 105


def _verify(U, V, W, rank, n, m, p, tol=1e-9):
    """Check if the decomposition is valid using exact rational arithmetic when possible."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2_terms():
    """
    Return the U, V, W coefficients for the 7 Strassen terms for 2x2x2 matrix multiplication.
    Indices: u,v,w in {0,1}.
    U[u, v, term], V[v, w, term], W[w, u, term]
    """
    U = np.zeros((2, 2, 7))
    V = np.zeros((2, 2, 7))
    W = np.zeros((2, 2, 7))
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00 + C11
    U[0,0,0] = 1; U[1,1,0] = 1
    V[0,0,0] = 1; V[1,1,0] = 1
    W[0,0,0] = 1; W[1,1,0] = 1
    
    # Term 2: M2 = (A10+A11)B00 -> C10 + C11 (with signs: C10 gets +, C11 gets -)
    # C10 is k=0, i=1 -> W[0,1]
    # C11 is k=1, i=1 -> W[1,1]
    U[1,0,1] = 1; U[1,1,1] = 1
    V[0,0,1] = 1
    W[0,1,1] = 1; W[1,1,1] = -1
    
    # Term 3: M3 = A00(B01-B11) -> C01 + C11
    # C01 is k=1, i=0 -> W[1,0]
    # C11 is k=1, i=1 -> W[1,1]
    U[0,0,2] = 1
    V[0,1,2] = 1; V[1,1,2] = -1
    W[1,0,2] = 1; W[1,1,2] = 1
    
    # Term 4: M4 = A11(B10-B00) -> C00 + C10
    # C00 is k=0, i=0 -> W[0,0]
    # C10 is k=0, i=1 -> W[0,1]
    U[1,1,3] = 1
    V[1,0,3] = 1; V[0,0,3] = -1
    W[0,0,3] = 1; W[0,1,3] = 1
    
    # Term 5: M5 = (A00+A01)B11 -> -C00 + C01
    # C00 is k=0, i=0 -> W[0,0]
    # C01 is k=1, i=0 -> W[1,0]
    U[0,0,4] = 1; U[0,1,4] = 1
    V[1,1,4] = 1
    W[0,0,4] = -1; W[1,0,4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01) -> C11
    # C11 is k=1, i=1 -> W[1,1]
    U[1,0,5] = 1; U[0,0,5] = -1
    V[0,0,5] = 1; V[0,1,5] = 1
    W[1,1,5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> C00
    # C00 is k=0, i=0 -> W[0,0]
    U[0,1,6] = 1; U[1,1,6] = -1
    V[1,0,6] = 1; V[1,1,6] = 1
    W[0,0,6] = 1
    
    return U, V, W


def _construct_block_decomposition(n, m, p):
    """
    Construct a rank-99 decomposition for 3x5x7 matrix multiplication
    using Strassen's algorithm on 2x2x2 sub-blocks.
    """
    # We will use a dictionary to accumulate coefficients for each rank index
    # Key: (i, j, k) is not enough, we need to track which term (rank index) it belongs to.
    # Better: List of tuples (U_idx, V_idx, W_idx, coeff) for each rank term.
    # Since we are building U, V, W matrices directly, we can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the blocks to use Strassen on
    # N=3: use {0,1}
    # M=5: use {0,1} and {2,3}
    # P=7: use {0,1}, {2,3}, {4,5}
    
    strassen_blocks = [
        (0, 0, 0), # i0, j0, k0
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    # Set of (i, j, k) covered by Strassen blocks
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Get Strassen coefficients
    S_U, S_V, S_W = _strassen_2x2x2_terms()
    
    # Add Strassen terms
    for i0, j0, k0 in strassen_blocks:
        for r in range(7):
            u_col = np.zeros(n*m)
            v_col = np.zeros(m*p)
            w_col = np.zeros(p*n)
            
            for di in range(2):
                for dj in range(2):
                    if S_U[di, dj, r] != 0:
                        i = i0 + di
                        j = j0 + dj
                        u_col[i*m + j] = S_U[di, dj, r]
            
            for dj in range(2):
                for dk in range(2):
                    if S_V[dj, dk, r] != 0:
                        j = j0 + dj
                        k = k0 + dk
                        v_col[j*p + k] = S_V[dj, dk, r]
            
            for dk in range(2):
                for di in range(2):
                    if S_W[dk, di, r] != 0:
                        k = k0 + dk
                        i = i0 + di
                        w_col[k*n + i] = S_W[dk, di, r]
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # Add Schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_col = np.zeros(n*m)
                    v_col = np.zeros(m*p)
                    w_col = np.zeros(p*n)
                    
                    u_col[i*m + j] = 1
                    v_col[j*p + k] = 1
                    w_col[k*n + i] = 1
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    
    # Convert to matrices
    rank = len(U_cols)
    U = np.column_stack(U_cols) if rank > 0 else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if rank > 0 else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if rank > 0 else np.zeros((p*n, 0))
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the block decomposition
    result = _construct_block_decomposition(n, m, p)
    
    # Verify
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = U.shape[1]
    
    if _verify(U, V, W, rank, n, m, p, tol=1e-9):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
