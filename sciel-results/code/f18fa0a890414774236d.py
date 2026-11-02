"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _to_list(mat):
    """Convert numpy object array to nested lists."""
    return mat.tolist()

def _check_exact(U, V, W, n, m, p):
    """
    Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k).
    Uses exact arithmetic (object dtype with ints/Fractions).
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition of rank n*m*p.
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with one 1 per term.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Strassen's 7-multiplication algorithm for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with dtype=object.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = -1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition for n x m x p matrix multiplication.
    Splits m and p into sub-blocks and uses Strassen where applicable.
    Returns (U, V, W, total_rank).
    """
    # For 3x5x6, split m=5 into 3+2, p=6 into 3+3
    # Sub-blocks:
    # (3,3,3), (3,3,3), (3,2,3), (3,2,3)
    # Use Strassen for (3,3,3) blocks? No, Strassen is for 2x2x2.
    # We need a decomposition for each sub-block.
    
    # Strategy: Use Strassen for 2x2 sub-blocks within the larger blocks.
    # For (3,3,3): split into 2x2 Strassen + extra terms.
    # Actually, let's use a general approach: for each sub-block, find the best
    # known decomposition.
    
    # For now, use schoolbook for all sub-blocks, but structure it so we can
    # later replace with Strassen for 2x2x2 sub-blocks.
    
    # Split: m = 3 + 2, p = 3 + 3
    m_splits = [3, 2]
    p_splits = [3, 3]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    total_rank = 0
    block_info = []  # (j_start, j_end, k_start, k_end, U, V, W, rank)
    
    for j_start, j_len in [(0, 3), (3, 2)]:
        for k_start, k_len in [(0, 3), (3, 3)]:
            # Sub-block: (n, j_len, k_len)
            # Use schoolbook for this sub-block
            U_b, V_b, W_b, r_b = _schoolbook(n, j_len, k_len)
            U_blocks.append(U_b)
            V_blocks.append(V_b)
            W_blocks.append(W_b)
            block_info.append((j_start, j_start + j_len, k_start, k_start + k_len, U_b, V_b, W_b, r_b))
            total_rank += r_b
    
    # Assemble full matrices
    U_full = np.zeros((n * m, total_rank), dtype=object)
    V_full = np.zeros((m * p, total_rank), dtype=object)
    W_full = np.zeros((p * n, total_rank), dtype=object)
    
    col_offset = 0
    for j_start, j_end, k_start, k_end, U_b, V_b, W_b, r_b in block_info:
        for c in range(r_b):
            # U: rows i*m + j for i in [0, n), j in [j_start, j_end)
            for i in range(n):
                for j in range(j_start, j_end):
                    u_idx = i * m + j
                    ub_i = i
                    ub_j = j - j_start
                    ub_idx = ub_i * (j_end - j_start) + ub_j
                    U_full[u_idx, col_offset + c] = U_b[ub_idx, c]
            
            # V: rows j*p + k for j in [j_start, j_end), k in [k_start, k_end)
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    v_idx = j * p + k
                    vb_j = j - j_start
                    vb_k = k - k_start
                    vb_idx = vb_j * (k_end - k_start) + vb_k
                    V_full[v_idx, col_offset + c] = V_b[vb_idx, c]
            
            # W: rows k*n + i for k in [k_start, k_end), i in [0, n)
            for k in range(k_start, k_end):
                for i in range(n):
                    w_idx = k * n + i
                    wb_k = k - k_start
                    wb_i = i
                    wb_idx = wb_k * n + wb_i
                    W_full[w_idx, col_offset + c] = W_b[wb_idx, c]
        
        col_offset += r_b
    
    return U_full, V_full, W_full, total_rank

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    Returns (U, V, W, new_rank) if reduction successful, else original.
    """
    r = U.shape[1]
    # Check for linear dependence among columns
    # A column c is redundant if there exist coefficients alpha_d (d != c) such that
    # for all i,j,k: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_d alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
    
    # This is complex. Instead, try a simpler approach:
    # If two columns are identical, remove one.
    # Or if a column is a scalar multiple of another, remove one.
    
    # For now, just check for exact duplicates
    seen = {}
    keep = []
    for c in range(r):
        # Create a signature for this column
        sig = (tuple(U[:, c].tolist()), tuple(V[:, c].tolist()), tuple(W[:, c].tolist()))
        if sig in seen:
            continue  # Skip duplicate
        seen[sig] = True
        keep.append(c)
    
    if len(keep) == r:
        return U, V, W, r
    
    # Remove duplicate columns
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    new_rank = len(keep)
    
    # Verify
    if _check_exact(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new, new_rank
    
    return U, V, W, r

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under 150s
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try block decomposition
    try:
        U_b, V_b, W_b, rank_b = _build_block_decomposition(n, m, p)
        if _check_exact(U_b, V_b, W_b, n, m, p) and rank_b < rank:
            U, V, W, rank = U_b, V_b, W_b, rank_b
    except Exception:
        pass
    
    # Try to reduce rank further
    if time.time() < deadline - 20:
        try:
            U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
        except Exception:
            pass
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
