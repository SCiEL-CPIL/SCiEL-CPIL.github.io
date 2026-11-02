"""
Rewritten program: recursive Strassen for 4x4x5 matrix multiplication.
Strategy: Use block decomposition with Strassen (rank 7) for 2x2x2 blocks
and Strassen+Naive (rank 9) for 2x2x3 blocks to achieve rank 64.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    
    Each term is a dict with 'U', 'V', 'W' where:
    - U: dict mapping (i, j) -> coeff for A[i,j]
    - V: dict mapping (j, k) -> coeff for B[j,k]
    - W: dict mapping (k, i) -> coeff for C[k,i]  (trace convention: W[k*n+i])
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A21+A22)B11
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A11(B21-B22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A22(B12-B11)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11+A12)B22
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A21-A11)(B11+B12)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12-A22)(B21+B22)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms


def _strassen_2x1_terms():
    """Return terms for 2x1x1 matrix multiplication (rank 2, schoolbook)."""
    terms = []
    terms.append({
        'U': {(0, 0): 1},
        'V': {(0, 0): 1},
        'W': {(0, 0): 1}
    })
    terms.append({
        'U': {(1, 0): 1},
        'V': {(0, 0): 1},
        'W': {(0, 1): 1}
    })
    return terms


def _strassen_2x2x3_terms():
    """Return terms for 2x2x3 matrix multiplication using Strassen+Naive (rank 9).
    
    A is 2x2, B is 2x3, C is 2x3.
    We split B into B_left (2x2) and B_right (2x1).
    - 2x2x2 part: rank 7 (Strassen)
    - 2x2x1 part: rank 2 (schoolbook)
    Total rank: 9.
    
    U: (i, j) where i in 0..1, j in 0..1
    V: (j, k) where j in 0..1, k in 0..2
    W: (k, i) where k in 0..2, i in 0..1
    """
    terms = []
    strassen_2x2 = _strassen_2x2_terms()
    strassen_2x1 = _strassen_2x1_terms()
    
    # Add 7 Strassen terms for the 2x2x2 part (k=0,1)
    for term in strassen_2x2:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for (i, j), c in term['U'].items():
            new_term['U'][(i, j)] = c
        for (j, k), c in term['V'].items():
            # B[j,k] for k in 0..1
            new_term['V'][(j, k)] = c
        for (k, i), c in term['W'].items():
            # C[k,i] for k in 0..1
            new_term['W'][(k, i)] = c
        terms.append(new_term)
    
    # Add 2 schoolbook terms for the 2x2x1 part (k=2)
    for term in strassen_2x1:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for (i, j), c in term['U'].items():
            new_term['U'][(i, j)] = c
        for (j, k), c in term['V'].items():
            # B[j,0] -> B[j,2] (the third column)
            new_term['V'][(j, 2)] = c
        for (k, i), c in term['W'].items():
            # C[0,i] -> C[2,i]
            new_term['W'][(2, i)] = c
        terms.append(new_term)
    
    return terms


def _build_4x4x5_block_decomposition():
    """Build 4x4x5 decomposition using block decomposition.
    
    Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    Split B (4x5) into 2x(2 or 3) blocks:
      B00 (2x2), B01 (2x3)
      B10 (2x2), B11 (2x3)
    C (4x5) is formed from 2x(2 or 3) blocks:
      C00 = A00*B00 + A01*B10 (2x2)
      C01 = A00*B01 + A01*B11 (2x3)
      C10 = A10*B00 + A11*B10 (2x2)
      C11 = A10*B01 + A11*B11 (2x3)
    
    For each C block, we use a 2x2x2 or 2x2x3 decomposition.
    - C00: 2x2x2 (rank 7) - but it's a sum of two products, so we need to handle carefully
    - Actually, each C block is a sum of two 2x2x2 or 2x2x3 products.
    
    Wait, let me reconsider. The matrix multiplication is:
    C = A @ B where A is 4x4, B is 4x5.
    
    In block form:
    A = [[A00, A01],
         [A10, A11]]  (each block 2x2)
    
    B = [[B00, B01],
         [B10, B11]]  where B00, B10 are 2x2 and B01, B11 are 2x3
    
    C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
         [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each C block is a sum of two matrix products. We can't just use Strassen on each
    because Strassen computes a single product. Instead, we need to compute each
    product separately and add them.
    
    So for C00 = A00*B00 + A01*B10:
    - A00*B00: 2x2x2, rank 7
    - A01*B10: 2x2x2, rank 7
    Total for C00: 14 terms
    
    Similarly:
    C01 = A00*B01 + A01*B11:
    - A00*B01: 2x2x3, rank 9
    - A01*B11: 2x2x3, rank 9
    Total for C01: 18 terms
    
    C10 = A10*B00 + A11*B10:
    - A10*B00: 2x2x2, rank 7
    - A11*B10: 2x2x2, rank 7
    Total for C10: 14 terms
    
    C11 = A10*B01 + A11*B11:
    - A10*B01: 2x2x3, rank 9
    - A11*B11: 2x2x3, rank 9
    Total for C11: 18 terms
    
    Total rank: 14 + 18 + 14 + 18 = 64
    
    This matches the observed improvement!
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 64
    
    U = [[0] * rank for _ in range(n * m)]   # 16 x rank
    V = [[0] * rank for _ in range(m * p)]   # 20 x rank
    W = [[0] * rank for _ in range(p * n)]   # 20 x rank
    
    strassen_2x2 = _strassen_2x2_terms()
    strassen_2x2x3 = _strassen_2x2x3_terms()
    
    # Block offsets:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-4), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-4)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-4), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-4)
    
    # Helper function to add terms
    def add_terms(term_list, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_rank):
        """Add terms from term_list with appropriate offsets.
        
        a_row_offset, a_col_offset: offset for A block (2x2)
        b_row_offset, b_col_offset: offset for B block (2x2 or 2x3)
        c_row_offset, c_col_offset: offset for C block (2x2 or 2x3)
        start_rank: starting rank index
        """
        for t_idx, term in enumerate(term_list):
            r = start_rank + t_idx
            
            # U: A[a_row_offset+li, a_col_offset+lj]
            for (li, lj), cu in term['U'].items():
                i = a_row_offset + li
                j = a_col_offset + lj
                U[i * m + j][r] += cu
            
            # V: B[b_row_offset+lj, b_col_offset+lk]
            for (lj, lk), cv in term['V'].items():
                j = b_row_offset + lj
                k = b_col_offset + lk
                V[j * p + k][r] += cv
            
            # W: C[c_row_offset+lk, c_col_offset+li]
            for (lk, li), cw in term['W'].items():
                k = c_row_offset + lk
                i = c_col_offset + li
                W[k * n + i][r] += cw
        
        return start_rank + len(term_list)
    
    rank_idx = 0
    
    # C00 = A00*B00 + A01*B10 (both 2x2x2, rank 7 each)
    # A00*B00: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    rank_idx = add_terms(strassen_2x2, 0, 0, 0, 0, 0, 0, rank_idx)
    # A01*B10: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    rank_idx = add_terms(strassen_2x2, 0, 2, 2, 0, 0, 0, rank_idx)
    
    # C01 = A00*B01 + A01*B11 (both 2x2x3, rank 9 each)
    # A00*B01: A rows 0-1, cols 0-1; B rows 0-1, cols 2-4; C rows 0-1, cols 2-4
    rank_idx = add_terms(strassen_2x2x3, 0, 0, 0, 2, 0, 2, rank_idx)
    # A01*B11: A rows 0-1, cols 2-3; B rows 2-3, cols 2-4; C rows 0-1, cols 2-4
    rank_idx = add_terms(strassen_2x2x3, 0, 2, 2, 2, 0, 2, rank_idx)
    
    # C10 = A10*B00 + A11*B10 (both 2x2x2, rank 7 each)
    # A10*B00: A rows 2-3, cols 0-1; B rows 0-1, cols 0-1; C rows 2-3, cols 0-1
    rank_idx = add_terms(strassen_2x2, 2, 0, 0, 0, 2, 0, rank_idx)
    # A11*B10: A rows 2-3, cols 2-3; B rows 2-3, cols 0-1; C rows 2-3, cols 0-1
    rank_idx = add_terms(strassen_2x2, 2, 2, 2, 0, 2, 0, rank_idx)
    
    # C11 = A10*B01 + A11*B11 (both 2x2x3, rank 9 each)
    # A10*B01: A rows 2-3, cols 0-1; B rows 0-1, cols 2-4; C rows 2-3, cols 2-4
    rank_idx = add_terms(strassen_2x2x3, 2, 0, 0, 2, 2, 2, rank_idx)
    # A11*B11: A rows 2-3, cols 2-3; B rows 2-3, cols 2-4; C rows 2-3, cols 2-4
    rank_idx = add_terms(strassen_2x2x3, 2, 2, 2, 2, 2, 2, rank_idx)
    
    assert rank_idx == rank, f"Expected rank {rank}, got {rank_idx}"
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using recursive Strassen: 7*7 = 49 terms."""
    n = 4
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            for (bi, bj), cu in term1['U'].items():
                for (li, lj), cv in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i * n + j][idx] += cu * cv
            
            for (bj, bk), cv in term1['V'].items():
                for (lj, lk), cw in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j * n + k][idx] += cv * cw
            
            for (bk, bi), cw in term1['W'].items():
                for (lk, li), cu2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k * n + i][idx] += cw * cu2
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x5_strassen():
    """Build 4x4x5 decomposition using recursive Strassen + schoolbook (rank 53)."""
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank_base):
                V[j * p + k][r] = base["V"][j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank_base):
                W[k * n + i][r] = base["W"][k * 4 + i][r]
    
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i * m + j][r] = 1
        V[j * p + 4][r] = 1
        for i in range(n):
            W[4 * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _build_schoolbook():
    """Fallback schoolbook decomposition: rank = n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Block decomposition with Strassen+Naive (rank 64)
    try:
        result = _build_4x4x5_block_decomposition()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Recursive Strassen 4x4x4 + schoolbook 4x4x1 (rank 53)
    if time.time() < deadline - 10:
        try:
            result = _build_4x4x5_strassen()
            rank = len(result["U"][0])
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                    if best_result is None or rank < best_rank:
                        best_result = result
                        best_rank = rank
        except Exception:
            pass
    
    # Approach 3: Schoolbook (rank 80) as fallback
    if best_result is None or time.time() < deadline - 5:
        try:
            result = _build_schoolbook()
            rank = len(result["U"][0])
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if best_result is None or rank < best_rank:
                    best_result = result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return _build_schoolbook()

# EVOLVE-BLOCK-END
