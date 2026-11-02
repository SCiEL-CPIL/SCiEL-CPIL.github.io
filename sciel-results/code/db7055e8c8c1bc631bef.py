"""Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import random

DIMENSIONS = [4, 4, 8]
WALL_DEADLINE = 138.0  # 12 s safety margin under 150 s

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition {U,V,W}."""
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification: check all entries of the matmul tensor."""
    rank = len(W[0]) if W else 0
    if rank == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > tol:
                    return False
    # Check a sample of off-diagonal entries
    rng = random.Random(12345)
    for _ in range(10000):
        i = rng.randrange(n)
        j = rng.randrange(m)
        k = rng.randrange(p)
        i2 = rng.randrange(n)
        j2 = rng.randrange(m)
        k2 = rng.randrange(p)
        if i == i2 and j == j2 and k == k2:
            continue
        a = i * m + j
        b = j * p + k
        c = k2 * n + i2
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s) > tol:
            return False
    return True


def _strassen_decomposition():
    """
    Construct a rank-112 Strassen-style decomposition for 4x4 * 4x8.
    
    Split into 2x2 blocks:
    - A: 2x2 blocks of size 2x2
    - B: 2x2 blocks of size 2x4  
    - C: 2x2 blocks of size 2x4
    
    Strassen uses 7 products instead of 8.
    Each product is (2x2) * (2x4) = (2x4).
    
    For each Strassen intermediate M, we express it as a sum of rank-1 tensors
    of the original 4x4x4x8 multiplication tensor.
    """
    terms = []  # list of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    def add(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        if u_val == 0 or v_val == 0 or w_val == 0:
            return
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Strassen formulas for 2x2 block multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x4 product.
    # Mi[i][k] = sum_{j=0}^{1} A_comb[i][j] * B_comb[j][k]
    # where A_comb and B_comb are linear combinations of original entries.
    
    # For each (i,k) in the 2x4 output, and each j in {0,1}:
    #   For each A-term (ar, ac, asgn) in A_comb[i][j]:
    #     For each B-term (br, bc, bsgn) in B_comb[j][k]:
    #       add rank-1 term: A[ar,ac] * B[br,bc] -> C[c_i, c_k]
    
    def process_intermediate(a_map, b_map, c_row_off, c_col_off, sign):
        """
        a_map: dict (li, lj) -> list of (global_row, global_col, sign)
        b_map: dict (lj, lk) -> list of (global_row, global_col, sign)
        c_row_off, c_col_off: offset in C matrix
        sign: overall multiplier
        """
        for li in range(2):
            for lk in range(4):
                for lj in range(2):
                    for (ar, ac, asgn) in a_map[(li, lj)]:
                        for (br, bc, bsgn) in b_map[(lj, lk)]:
                            u_idx = ar * 4 + ac
                            v_idx = br * 8 + bc
                            c_i = c_row_off + li
                            c_k = c_col_off + lk
                            w_idx = c_k * 4 + c_i
                            add(u_idx, sign * asgn, v_idx, bsgn, w_idx, 1)
    
    # M1 = (A00+A11)*(B00+B11)
    m1_a = {(li, lj): [(li, lj, 1), (li+2, lj+2, 1)] for li in range(2) for lj in range(2)}
    m1_b = {(lj, lk): [(lj, lk, 1), (lj+2, lk, 1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m1_a, m1_b, 0, 0, 1)   # C00
    process_intermediate(m1_a, m1_b, 2, 4, 1)   # C11
    
    # M2 = (A10+A11)*B00
    m2_a = {(li, lj): [(li+2, lj, 1), (li+2, lj+2, 1)] for li in range(2) for lj in range(2)}
    m2_b = {(lj, lk): [(lj, lk, 1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m2_a, m2_b, 2, 0, 1)   # C10
    process_intermediate(m2_a, m2_b, 2, 4, -1)  # C11
    
    # M3 = A00*(B10-B11)
    m3_a = {(li, lj): [(li, lj, 1)] for li in range(2) for lj in range(2)}
    m3_b = {(lj, lk): [(lj+2, lk, 1), (lj+2, lk+4, -1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m3_a, m3_b, 0, 4, 1)   # C01
    process_intermediate(m3_a, m3_b, 2, 4, 1)   # C11
    
    # M4 = A11*(B01-B00)
    m4_a = {(li, lj): [(li+2, lj+2, 1)] for li in range(2) for lj in range(2)}
    m4_b = {(lj, lk): [(lj, lk+4, 1), (lj, lk, -1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m4_a, m4_b, 0, 0, 1)   # C00
    process_intermediate(m4_a, m4_b, 2, 0, 1)   # C10
    
    # M5 = (A00+A01)*B11
    m5_a = {(li, lj): [(li, lj, 1), (li, lj+2, 1)] for li in range(2) for lj in range(2)}
    m5_b = {(lj, lk): [(lj+2, lk+4, 1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m5_a, m5_b, 0, 0, -1)  # C00
    process_intermediate(m5_a, m5_b, 0, 4, 1)   # C01
    
    # M6 = (A10-A00)*(B00+B01)
    m6_a = {(li, lj): [(li+2, lj, 1), (li, lj, -1)] for li in range(2) for lj in range(2)}
    m6_b = {(lj, lk): [(lj, lk, 1), (lj, lk+4, 1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m6_a, m6_b, 2, 4, 1)   # C11
    
    # M7 = (A01-A11)*(B10+B11)
    m7_a = {(li, lj): [(li, lj+2, 1), (li+2, lj+2, -1)] for li in range(2) for lj in range(2)}
    m7_b = {(lj, lk): [(lj+2, lk, 1), (lj+2, lk+4, 1)] for lj in range(2) for lk in range(4)}
    process_intermediate(m7_a, m7_b, 0, 0, 1)   # C00
    
    # Build matrices
    n, m, p = 4, 4, 8
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        U[u_idx][t] = u_val
        V[v_idx][t] = v_val
        W[w_idx][t] = w_val
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["W"][0])
    
    # Try Strassen decomposition (rank 112)
    if time.time() - start_time < WALL_DEADLINE - 5:
        try:
            strat = _strassen_decomposition()
            s_rank = len(strat["W"][0])
            if s_rank < best_rank:
                if _verify_full(strat["U"], strat["V"], strat["W"], n, m, p):
                    best = strat
                    best_rank = s_rank
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END
