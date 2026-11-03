"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 is W[0], C11 is W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 is W[1], C11 is W[3]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # A00 is U[0]
    # B01 is V[1], B11 is V[3]
    # C01 is W[2], C11 is W[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C00 is W[0], C10 is W[1]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C00 is W[0], C01 is W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B01 is V[1]
    # C11 is W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    # A01 is U[1], A11 is U[3]
    # B10 is V[2], B11 is V[3]
    # C00 is W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_block_solution(n, m, p):
    """Build the block-structured Strassen/schoolbook solution (rank 50)."""
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]

    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()

    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                p_sub = 1

            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]

            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]

            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]

            col_offset += r_sub

    return U, V, W

def _greedy_sparse_search(n, m, p, max_rank, deadline):
    """
    Greedy sparse support search for a low-rank decomposition.
    We look for rank-1 terms (u, v, w) with integer/half-integer entries
    that together satisfy the tensor identity.
    
    The tensor identity is:
      sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    
    We use a greedy approach: start with the block solution, then try to
    merge/simplify columns to reduce rank.
    
    Alternative: build from scratch by greedily adding rank-1 terms that
    satisfy the most remaining equations.
    """
    # First, let's try a direct greedy construction.
    # The tensor has n*m*p = 2*4*7 = 56 "trivial" terms.
    # We want to find a set of rank-1 terms that satisfies all 140 equations
    # (n * m * m * p = 2*4*4*7 = 224... wait let me recount)
    # Actually the identity is: for each (i, j, j', k), sum over c of 
    # W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    # Number of equations: n * m * m * p = 2 * 4 * 4 * 7 = 224
    # But delta_{j,j'} means only m^2 * n * p = 4*2*7 = 56 are non-trivial
    # (the others must be 0).
    
    # Let's try a different approach: use the block solution as a starting point
    # and see if we can reduce the rank by finding linear dependencies.
    
    U0, V0, W0 = _build_block_solution(n, m, p)
    r0 = U0.shape[1]
    
    # Try to find a lower rank by solving the linear system directly.
    # We'll use a greedy approach to select which rank-1 terms to include.
    
    # Actually, let's try a simpler approach: verify the block solution works
    # and return it. The greedy search for a strictly lower rank is complex
    # and may not succeed within the time budget.
    
    if time.time() > deadline:
        return None
    
    # Verify the block solution
    if _verify(U0, V0, W0, n, m, p):
        return (U0, V0, W0)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Try the block-structured approach first
    result = _greedy_sparse_search(n, m, p, 50, deadline)
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    r = n * m * p
    U_fb = [[0] * r for _ in range(n * m)]
    V_fb = [[0] * r for _ in range(m * p)]
    W_fb = [[0] * r for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
