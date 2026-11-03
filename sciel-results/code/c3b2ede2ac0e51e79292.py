"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W give a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                if is_valid:
                    if abs(s - 1.0) > tol:
                        return False
                else:
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u_indices, v_indices, w_indices) where:
      u_indices: list of (row, col) in the 2x2 A-block
      v_indices: list of (row, col) in the 2x2 B-block  
      w_indices: list of (row, col) in the 2x2 C-block
    with +/- coefficients.
    
    Strassen's 7 products for C = A*B where A,B,C are 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each product: (coeff_A, coeff_B, coeff_C)
    # where coeff is a dict mapping (row,col) -> coefficient
    terms = []
    
    # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    terms.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
    terms.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = A11(B21-B22) -> C12 += 1, C22 += 1
    terms.append({
        'A': {(0,0): 1},
        'B': {(1,0): 1, (1,1): -1},
        'C': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A22(B12-B11) -> C11 += 1, C21 += 1
    terms.append({
        'A': {(1,1): 1},
        'B': {(0,1): 1, (0,0): -1},
        'C': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
    terms.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (A21-A11)(B11+B12) -> C22 += 1
    terms.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1}
    })
    
    # M7 = (A12-A22)(B21+B22) -> C11 += 1, C22 += 1
    terms.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    })
    
    return terms

def _build_decomposition(n, m, p):
    """
    Build a decomposition for n x m times m x p matrix multiplication.
    
    Strategy: Split the m dimension into blocks of size 2 where possible.
    For 3x4x4: 
      - Row 0 of A (1x4) times B (4x4): use naive, rank = 1*4*4 = 16
      - Rows 1-2 of A (2x4) times B (4x4): use Strassen on 2x2 blocks, rank = 2*7*2 = 28
      Total rank = 16 + 28 = 44
    
    More generally, we partition the m columns into blocks.
    For each row block of A and column block of B, we can apply Strassen
    if both are 2x2 submatrices.
    """
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    rank = 0
    strassen_terms = _strassen_2x2_terms()
    
    # We need to partition:
    # - Rows of A (n rows) into blocks
    # - Columns of A / rows of B (m) into blocks  
    # - Columns of B (p) into blocks
    
    # For 3x4x4:
    # A is 3x4, B is 4x4, C is 4x3
    # Split A's columns (m=4) into two blocks of 2: cols 0-1, cols 2-3
    # Split B's rows (m=4) into two blocks of 2: rows 0-1, rows 2-3
    # Split B's columns (p=4) into two blocks of 2: cols 0-1, cols 2-3
    
    # For each (row_block_of_A, col_block_of_A_and_row_block_of_B, col_block_of_B):
    # If row_block is size 2 and the others are size 2, use Strassen (7 terms)
    # Otherwise use naive (size_a * size_b * size_c terms)
    
    # Let's define the partitions:
    # A rows: [0], [1,2]  (sizes 1, 2)
    # A cols / B rows: [0,1], [2,3]  (sizes 2, 2)
    # B cols: [0,1], [2,3]  (sizes 2, 2)
    
    a_row_blocks = [(0, 1), (1, 3)]  # (start, end) exclusive
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [(0, 2), (2, 4)]
    
    for a_rs, a_re in a_row_blocks:
        a_block_size = a_re - a_rs
        for m_s, m_e in m_blocks:
            m_block_size = m_e - m_s
            for p_s, p_e in p_blocks:
                p_block_size = p_e - p_s
                
                if a_block_size == 2 and m_block_size == 2 and p_block_size == 2:
                    # Use Strassen: 7 terms
                    for term in strassen_terms:
                        col = rank
                        # Add entries to U, V, W
                        for (ar, ac), coeff in term['A'].items():
                            a_idx = (a_rs + ar) * m + (m_s + ac)
                            if a_idx < len(U):
                                U[a_idx].append(coeff)
                        for (br, bc), coeff in term['B'].items():
                            b_idx = (m_s + br) * p + (p_s + bc)
                            if b_idx < len(V):
                                V[b_idx].append(coeff)
                        for (cr, cc), coeff in term['C'].items():
                            c_idx = (p_s + cr) * n + (a_rs + cc)
                            if c_idx < len(W):
                                W[c_idx].append(coeff)
                        rank += 1
                else:
                    # Naive: a_block_size * m_block_size * p_block_size terms
                    for ar in range(a_block_size):
                        for mr in range(m_block_size):
                            for pr in range(p_block_size):
                                col = rank
                                a_idx = (a_rs + ar) * m + (m_s + mr)
                                b_idx = (m_s + mr) * p + (p_s + pr)
                                c_idx = (p_s + pr) * n + (a_rs + ar)
                                if a_idx < len(U):
                                    U[a_idx].append(1)
                                if b_idx < len(V):
                                    V[b_idx].append(1)
                                if c_idx < len(W):
                                    W[c_idx].append(1)
                                rank += 1
    
    # Pad any rows that don't have enough columns
    for i in range(n * m):
        while len(U[i]) < rank:
            U[i].append(0)
    for i in range(m * p):
        while len(V[i]) < rank:
            V[i].append(0)
    for i in range(p * n):
        while len(W[i]) < rank:
            W[i].append(0)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the Strassen-based decomposition
    if time.time() < deadline:
        candidate = _build_decomposition(n, m, p)
        cand_rank = len(candidate["U"][0]) if candidate["U"] else 0
        
        # Verify
        if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            if cand_rank < best_rank:
                best = candidate
                best_rank = cand_rank
    
    if time.time() < deadline:
        # Try to verify schoolbook too (sanity check)
        if not _verify(best["U"], best["V"], best["W"], n, m, p):
            # Fallback to schoolbook if verification fails
            best = _schoolbook(n, m, p)
    
    return best
# EVOLVE-BLOCK-END
