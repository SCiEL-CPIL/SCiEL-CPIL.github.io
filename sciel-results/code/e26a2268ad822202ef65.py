"""Rewritten program: recursive Strassen for 4x4x5 matrix multiplication."""
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
    
    For 2x2: C = A @ B, C[i,k] = sum_j A[i,j]*B[j,k]
    The tensor is T[i,j,k] = 1, decomposed as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
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
    """Return terms for 2x1x1 matrix multiplication (rank 2, schoolbook).
    
    A is 2x1, B is 1x1, C is 2x1.
    C[0,0] = A[0,0]*B[0,0]
    C[1,0] = A[1,0]*B[0,0]
    
    U: (i, j) where i in 0..1, j in 0 (only j=0)
    V: (j, k) where j=0, k=0 (only one entry)
    W: (k, i) where k=0, i in 0..1
    """
    terms = []
    # Term 1: A[0,0] * B[0,0] -> C[0,0]
    terms.append({
        'U': {(0, 0): 1},
        'V': {(0, 0): 1},
        'W': {(0, 0): 1}
    })
    # Term 2: A[1,0] * B[0,0] -> C[1,0]
    terms.append({
        'U': {(1, 0): 1},
        'V': {(0, 0): 1},
        'W': {(0, 1): 1}
    })
    return terms


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using recursive Strassen: 7*7 = 49 terms.
    
    The 4x4x4 multiplication is decomposed into 7 outer Strassen terms,
    each of which involves a 2x2x2 multiplication, which itself uses 7 terms.
    Total rank: 49.
    
    U: shape (16, 49) - indexed by (i, j) for A[i,j], i,j in 0..3
    V: shape (16, 49) - indexed by (j, k) for B[j,k], j,k in 0..3
    W: shape (16, 49) - indexed by (k, i) for C[k,i], k,i in 0..3 (trace convention)
    """
    n = 4
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            # U: A is 4x4, block structure: A[bi*2+li, bj*2+lj]
            # term1['U'] gives the outer 2x2 block selection (bi, bj)
            # term2['U'] gives the inner 2x2 element selection (li, lj)
            for (bi, bj), cu in term1['U'].items():
                for (li, lj), cv in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i * n + j][idx] += cu * cv
            
            # V: B is 4x4, B[bj*2+lj, bk*2+lk]
            # term1['V'] gives outer (bj, bk)
            # term2['V'] gives inner (lj, lk)
            for (bj, bk), cv in term1['V'].items():
                for (lj, lk), cw in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j * n + k][idx] += cv * cw
            
            # W: C is 4x4, C[bk*2+lk, bi*2+li] (trace convention: W[k*n+i])
            # term1['W'] gives outer (bk, bi)
            # term2['W'] gives inner (lk, li)
            for (bk, bi), cw in term1['W'].items():
                for (lk, li), cu2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k * n + i][idx] += cw * cu2
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x5_strassen():
    """Build 4x4x5 decomposition using recursive Strassen.
    
    Strategy: Split B (4x5) into B_left (4x4) and B_right (4x1).
    - 4x4x4 part: rank 49 (recursive Strassen)
    - 4x4x1 part: rank 4 (schoolbook for the last column)
    Total rank: 53.
    
    Alternatively, we can try a more clever decomposition.
    
    Actually, let's think about this differently. We can split the problem:
    A is 4x4, B is 4x5, C is 4x5.
    
    Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (2 rows x 2-3 cols).
    
    Better approach: Use the recursive Strassen structure on the 4x4 part,
    and handle the extra column separately.
    
    For the 4x4x1 part (C[:, 4] = A @ B[:, 4]):
    This is a 4x4 times 4x1 multiplication. We can use 4 terms (schoolbook)
    or potentially fewer with Strassen-like tricks.
    
    Actually, for 4x4x1: C[i,0] = sum_j A[i,j] * B[j,0]
    This is just a vector-matrix product. The minimum rank is 4 (you need at
    least 4 terms since each output C[i,0] depends on all 4 inputs A[i,j]).
    
    Wait, actually the rank of the 4x4x1 tensor is exactly 4. Each term r
    contributes U[i,j,r]*V[j,0,r]*W[0,i,r] to C[i,0].
    Since V only has one column (j=0..3, k=0), V[j,0,r] for j=0..3.
    W[0,i,r] for i=0..3.
    U[i,j,r] for i,j=0..3.
    
    The tensor T[i,j,0] = 1 for all i,j.
    We need sum_r U[i,j,r] * V[j,0,r] * W[0,i,r] = 1 for all i,j.
    
    With rank 4: U[i,j,r] = delta(i,r), V[j,0,r] = 1, W[0,i,r] = delta(i,r)? No...
    Let's just use: for each j, one term where U[i,j,r_j]=1 for all i, V[j,0,r_j]=1, W[0,i,r_j]=1.
    That's 4 terms.
    
    So total rank = 49 + 4 = 53.
    
    But wait - can we do better by not separating them? Let's think...
    
    Actually, there's a smarter approach. Instead of treating B as 4x4 + 4x1,
    we can use the full 4x5 structure with Strassen on the 4x4 blocks and handle
    the fifth column more efficiently.
    
    Let me try another approach: split into 2x2x2 and 2x2x1 blocks.
    A: 4x4 -> 4 blocks of 2x2
    B: 4x5 -> 2 rows of 2x(2 or 3)
    
    This is getting complex. Let me stick with the simple 49+4=53 approach
    and verify it works.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build 4x4x4 part (first 4 columns of B, first 4 columns of C)
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]   # 16 x rank
    V = [[0] * rank for _ in range(m * p)]   # 20 x rank
    W = [[0] * rank for _ in range(p * n)]   # 20 x rank
    
    # Copy 4x4x4 part into first 49 columns
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):  # j*p index for first 4 cols of B: j*5+k, k=0..3
        for r in range(rank_base):
            # V is indexed by (j, k) -> V[j*p + k]
            # base["V"] is indexed by (j, k) for 4x4, so V[j*4+k]
            # We need to map: base V[j*4+k] -> our V[j*5+k] for k=0..3
            row_in_base = j  # this is j*4+k in base, but we iterate j as row
            # Actually base V has shape (16, 49), indexed by j*4+k
            # Our V has shape (20, rank), indexed by j*5+k
            # For the 4x4 part: base V[j*4+k] goes to our V[j*5+k] for k=0..3
            pass
    
    # Let me redo this more carefully
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4x4x4 part: A (4x4), B (4x4) -> C (4x4)
    # In our full problem: A is 4x4, B is 4x5, C is 4x5
    # The 4x4x4 part uses B[:, 0:4] and produces C[:, 0:4]
    
    for i in range(16):  # U rows: i*4+j, i,j in 0..3
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank_base):
                # base V[j*4+k] -> our V[j*5+k]
                V[j * p + k][r] = base["V"][j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank_base):
                # base W[k*4+i] -> our W[k*4+i] (same indexing since n=4)
                W[k * n + i][r] = base["W"][k * 4 + i][r]
    
    # 4x4x1 part: A (4x4), B (4x1) -> C (4x1)
    # C[i, 4] = sum_j A[i,j] * B[j, 4]
    # Schoolbook: 4 terms, one per j
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i * m + j][r] = 1  # U[i,j] = 1
        V[j * p + 4][r] = 1      # V[j,4] = 1
        for i in range(n):
            W[4 * n + i][r] = 1  # W[4,i] = 1 (k=4, so index = 4*4+i = 16+i)
    
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
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Recursive Strassen 4x4x4 + schoolbook 4x4x1 (rank 53)
    try:
        result = _build_4x4x5_strassen()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Schoolbook (rank 80) as fallback
    if best_result is None or time.time() < deadline - 10:
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
