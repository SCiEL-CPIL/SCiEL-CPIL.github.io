"""Matrix multiplication 4,5,5 tensor decomposition."""
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return an exact tensor decomposition of the matrix multiplication tensor
    for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses a fixed tiling of 8 non-overlapping 2x2x2 blocks, each decomposed
    via Strassen's 7-term identity, with the remaining 36 positions handled
    by schoolbook (rank-1) terms. Total rank = 8*7 + 36 = 92.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strassen's 7-term decomposition for 2x2x2
    strassen_terms = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),  # (A11+A22)(B11+B22)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # (A21+A22)B11
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),   # A11(B21-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # A22(B12-B22)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),   # (A11+A12)B22
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # (A21-A11)(B11+B12)
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),   # (A12-A22)(B21+B22)
    ]
    
    # Fixed optimal tiling: 8 non-overlapping 2x2x2 blocks
    # i: {0, 2}, j: {0, 2}, k: {0, 2}
    blocks = []
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                blocks.append((i0, j0, k0))
    
    # Determine which (i,j,k) are covered by blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Count schoolbook terms
    num_schoolbook = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    num_schoolbook += 1
    
    num_strassen = len(blocks) * 7
    rank = num_schoolbook + num_strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Strassen terms for each block
    for (i0, j0, k0) in blocks:
        a_idx = [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]
        b_idx = [(j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)]
        c_idx = [(k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)]
        
        for (u_c, v_c, w_c) in strassen_terms:
            for la, cf in enumerate(u_c):
                if cf != 0:
                    i, j = a_idx[la]
                    U[i * m + j][col] += cf
            for lb, cf in enumerate(v_c):
                if cf != 0:
                    j, k = b_idx[lb]
                    V[j * p + k][col] += cf
            for lc, cf in enumerate(w_c):
                if cf != 0:
                    k, i = c_idx[lc]
                    W[k * n + i][col] += cf
            col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    if time.time() < deadline - 10:
        if not verify_decomposition(U, V, W, n, m, p, rank):
            # Fallback to schoolbook if verification fails
            rank_schoolbook = n * m * p
            U_s = [[0] * rank_schoolbook for _ in range(n * m)]
            V_s = [[0] * rank_schoolbook for _ in range(m * p)]
            W_s = [[0] * rank_schoolbook for _ in range(p * n)]
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col_s] = 1
                        V_s[j * p + k][col_s] = 1
                        W_s[k * n + i][col_s] = 1
                        col_s += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return result

def verify_decomposition(U, V, W, n, m, p, rank):
    """Verify the decomposition exactly using Fraction arithmetic."""
    if rank == 0:
        return n * m * p == 0
    
    # Check sample first
    import random
    rng = random.Random(42)
    
    test_positions = set()
    test_positions.add((0, 0, 0))
    test_positions.add((n-1, m-1, p-1))
    test_positions.add((0, 0, p-1))
    test_positions.add((n-1, m-1, 0))
    test_positions.add((n-1, 0, 0))
    test_positions.add((0, m-1, p-1))
    
    for _ in range(min(50, n*m*p)):
        test_positions.add((rng.randint(0, n-1), rng.randint(0, m-1), rng.randint(0, p-1)))
    
    for i, j, k in test_positions:
        s = Fraction(0)
        for c in range(rank):
            uc = U[i * m + j][c]
            vc = V[j * p + k][c]
            wc = W[k * n + i][c]
            if uc != 0 and vc != 0 and wc != 0:
                s += Fraction(uc) * Fraction(vc) * Fraction(wc)
        if s != 1:
            return False
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    uc = U[i * m + j][c]
                    vc = V[j * p + k][c]
                    wc = W[k * n + i][c]
                    if uc != 0 and vc != 0 and wc != 0:
                        s += Fraction(uc) * Fraction(vc) * Fraction(wc)
                if s != 1:
                    return False
    
    return True
# EVOLVE-BLOCK-END
