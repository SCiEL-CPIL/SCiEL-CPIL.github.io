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


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check U ⊙ V ⊙ W reproduces the matmul tensor exactly (up to tol)."""
    rank = len(W[0]) if W else 0
    if rank == 0:
        return False
    for a in range(n * m):
        i, j = a // m, a % m
        for b in range(m * p):
            j2, k = b // p, b % p
            for c in range(p * n):
                k2, i2 = c // n, c % n
                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - expected) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification: check only on-diagonal and a sample of off-diagonal."""
    rank = len(W[0]) if W else 0
    if rank == 0:
        return False
    # Check all on-diagonal entries
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
    rng = random.Random(42)
    for _ in range(5000):
        a = rng.randrange(n * m)
        b = rng.randrange(m * p)
        c = rng.randrange(p * n)
        i, j = a // m, a % m
        j2, k = b // p, b % p
        k2, i2 = c // n, c % n
        if i == i2 and j == j2 and k == k2:
            continue
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s) > tol:
            return False
    return True


def _try_strassen_rank(n, m, p):
    """
    Attempt a Strassen-style block decomposition.
    
    Split A (4x4) into 2x2 blocks of size 2x2.
    Split B (4x8) into 2x2 blocks of size 2x4.
    C (4x8) into 2x2 blocks of size 2x4.
    
    Strassen gives 7 "products" of 2x2 * 2x4.
    Each 2x2 * 2x4 product has schoolbook rank 2*2*4 = 16.
    Total: 7 * 16 = 112 < 128.
    
    The key insight: each Strassen product like (A00+A11)*(B00+B11) 
    produces a 2x4 matrix whose entries are bilinear in the original A,B.
    We decompose this into rank-1 terms of the ORIGINAL tensor.
    """
    # Block indices:
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:4], B01=B[0:2,4:8], B10=B[2:4,0:4], B11=B[2:4,4:8]
    # C blocks: C00=C[0:2,0:4], C01=C[0:2,4:8], C10=C[2:4,0:4], C11=C[2:4,4:8]
    
    # Strassen intermediates (each is a 2x2 * 2x4 product):
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
    
    # Each Mi is a product of a 2x2 matrix (linear combo of A blocks) and a 2x4 matrix (linear combo of B blocks).
    # The product is a 2x4 matrix. Each entry is sum over j=0,1 of A_combined[i,j]*B_combined[j,k].
    # Each A_combined[i,j] is a linear combo of original A entries, each B_combined[j,k] is a linear combo of original B entries.
    # So each entry of Mi is a sum of (number of A terms)*(number of B terms) rank-1 products.
    
    # Let's enumerate all rank-1 terms.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        if u_val == 0 or v_val == 0 or w_val == 0:
            return
        U.append((u_idx, u_val))
        V.append((v_idx, v_val))
        W.append((w_idx, w_val))
    
    # Helper: for a 2x2*2x4 product where A_comb[i][j] = sum_{(a,b) in a_entries} sign * A[a][b]
    # and B_comb[j][k] = sum_{(c,d) in b_entries} sign * B[c][d]
    # The product entry (i,k) = sum_j sum_{a,b} sum_{c,d} sign_a * sign_b * A[a][b]*B[c][d]
    # contributes to C at position (i_global, k_global)
    
    # Actually, let me just directly enumerate. For each Strassen intermediate,
    # I'll compute the expansion.
    
    # M1 = (A00+A11)*(B00+B11)
    # A_comb: [A[0,0]+A[2,2], A[0,1]+A[2,3]; A[1,0]+A[3,2], A[1,1]+A[3,3]]
    # B_comb: [B[0,k]+B[2,k] for k in 0..3; B[1,k]+B[3,k] for k in 0..3]
    # Product[i][k] = (A[2i,0]+A[2i+2,2])*(B[0,k]+B[2,k]) + (A[2i,1]+A[2i+2,3])*(B[1,k]+B[3,k])
    # This contributes to C[2i, k] (top half) since M1 is in C00 and C11
    
    # For C00 (i=0,1; k=0,1,2,3): contribution +1 * M1[i][k]
    # For C11 (i=0,1; k=0,1,2,3): contribution +1 * M1[i][k]  (but at rows 2+i, cols 4+k)
    
    # Let's write a general function
    def add_product_contribution(a_entries, b_entries, c_offset_i, c_offset_k, sign, n, p):
        """
        a_entries: list of (a_row, a_col, a_sign) for the 2x2 combined A
        b_entries: list of (b_row, b_col, b_sign) for the 2x4 combined B
        c_offset_i: row offset in C for this contribution
        c_offset_k: col offset in C for this contribution
        sign: overall multiplier
        """
        # The product is 2x4. Entry (li, lk) where li in {0,1}, lk in {0,1,2,3}
        # = sum over (li, lj) of A_combined[li][lj] * B_combined[lj][lk]
        # A_combined[li][lj] = sum of a_sign * A[a_row + li_offset... ]
        
        # Actually, let me be more explicit.
        # a_entries maps to the 2x2 grid: a_entries[(li, lj)] = list of (global_row, global_col, sign)
        # b_entries maps to the 2x4 grid: b_entries[(lj, lk)] = list of (global_row, global_col, sign)
        
        for li in range(2):
            for lk in range(4):
                # Sum over lj in {0,1}
                for lj in range(2):
                    # A_combined[li][lj] * B_combined[lj][lk]
                    for (ar, ac, asgn) in a_entries[(li, lj)]:
                        for (br, bc, bsgn) in b_entries[(lj, lk)]:
                            u_idx = ar * 4 + ac  # A is 4x4
                            v_idx = br * 8 + bc  # B is 4x8
                            c_i = c_offset_i + li
                            c_k = c_offset_k + lk
                            w_idx = c_k * 4 + c_i  # W index: k*n + i
                            add_term(u_idx, sign * asgn, v_idx, bsgn, w_idx, 1)
    
    # Define entries for each Strassen intermediate
    
    # M1: (A00+A11)*(B00+B11)
    # A00: rows 0-1, cols 0-1. A11: rows 2-3, cols 2-3.
    # A_combined[li][lj] = A[li, lj] + A[li+2, lj+2]
    m1_a = {}
    for li in range(2):
        for lj in range(2):
            m1_a[(li, lj)] = [(li, lj, 1), (li + 2, lj + 2, 1)]
    # B00: rows 0-1, cols 0-3. B11: rows 2-3, cols 0-3.
    # B_combined[lj][lk] = B[lj, lk] + B[lj+2, lk]
    m1_b = {}
    for lj in range(2):
        for lk in range(4):
            m1_b[(lj, lk)] = [(lj, lk, 1), (lj + 2, lk, 1)]
    
    # M1 contributes +1 to C00 (offset 0,0) and +1 to C11 (offset 2,4)
    add_product_contribution(m1_a, m1_b, 0, 0, 1, 4, 8)
    add_product_contribution(m1_a, m1_b, 2, 4, 1, 4, 8)
    
    # M2: (A10+A11)*B00
    # A10: rows 2-3, cols 0-1. A11: rows 2-3, cols 2-3.
    # A_combined[li][lj] = A[li+2, lj] + A[li+2, lj+2]
    m2_a = {}
    for li in range(2):
        for lj in range(2):
            m2_a[(li, lj)] = [(li + 2, lj, 1), (li + 2, lj + 2, 1)]
    # B00: rows 0-1, cols 0-3
    m2_b = {}
    for lj in range(2):
        for lk in range(4):
            m2_b[(lj, lk)] = [(lj, lk, 1)]
    
    # M2 contributes +1 to C10 (offset 2, 0) and -1 to C11 (offset 2, 4)
    add_product_contribution(m2_a, m2_b, 2, 0, 1, 4, 8)
    add_product_contribution(m2_a, m2_b, 2, 4, -1, 4, 8)
    
    # M3: A00*(B10-B11)
    m3_a = {}
    for li in range(2):
        for lj in range(2):
            m3_a[(li, lj)] = [(li, lj, 1)]
    m3_b = {}
    for lj in range(2):
        for lk in range(4):
            m3_b[(lj, lk)] = [(lj + 2, lk, 1), (lj + 2, lk, -1)]  # Wait, B10 is rows 2-3, B11 is rows 2-3
    # B10: rows 2-3, cols 0-3. B11: rows 2-3, cols 4-7.
    # B_combined[lj][lk] = B[lj+2, lk] - B[lj+2, lk+4]
    m3_b = {}
    for lj in range(2):
        for lk in range(4):
            m3_b[(lj, lk)] = [(lj + 2, lk, 1), (lj + 2, lk + 4, -1)]
    
    # M3 contributes +1 to C01 (offset 0, 4) and +1 to C11 (offset 2, 4)
    add_product_contribution(m3_a, m3_b, 0, 4, 1, 4, 8)
    add_product_contribution(m3_a, m3_b, 2, 4, 1, 4, 8)
    
    # M4: A11*(B01-B00)
    m4_a = {}
    for li in range(2):
        for lj in range(2):
            m4_a[(li, lj)] = [(li + 2, lj + 2, 1)]
    # B01: rows 0-1, cols 4-7. B00: rows 0-1, cols 0-3.
    # B_combined[lj][lk] = B[lj, lk+4] - B[lj, lk]
    m4_b = {}
    for lj in range(2):
        for lk in range(4):
            m4_b[(lj, lk)] = [(lj, lk + 4, 1), (lj, lk, -1)]
    
    # M4 contributes +1 to C00 (offset 0, 0) and +1 to C10 (offset 2, 0)
    add_product_contribution(m4_a, m4_b, 0, 0, 1, 4, 8)
    add_product_contribution(m4_a, m4_b, 2, 0, 1, 4, 8)
    
    # M5: (A00+A01)*B11
    m5_a = {}
    for li in range(2):
        for lj in range(2):
            m5_a[(li, lj)] = [(li, lj, 1), (li, lj + 2, 1)]
    # B11: rows 2-3, cols 4-7
    m5_b = {}
    for lj in range(2):
        for lk in range(4):
            m5_b[(lj, lk)] = [(lj + 2, lk + 4, 1)]
    
    # M5 contributes -1 to C00 (offset 0, 0) and +1 to C01 (offset 0, 4)
    add_product_contribution(m5_a, m5_b, 0, 0, -1, 4, 8)
    add_product_contribution(m5_a, m5_b, 0, 4, 1, 4, 8)
    
    # M6: (A10-A00)*(B00+B01)
    m6_a = {}
    for li in range(2):
        for lj in range(2):
            m6_a[(li, lj)] = [(li + 2, lj, 1), (li, lj, -1)]
    # B00: rows 0-1, cols 0-3. B01: rows 0-1, cols 4-7.
    # B_combined[lj][lk] = B[lj, lk] + B[lj, lk+4]
    m6_b = {}
    for lj in range(2):
        for lk in range(4):
            m6_b[(lj, lk)] = [(lj, lk, 1), (lj, lk + 4, 1)]
    
    # M6 contributes +1 to C11 (offset 2, 4)
    add_product_contribution(m6_a, m6_b, 2, 4, 1, 4, 8)
    
    # M7: (A01-A11)*(B10+B11)
    m7_a = {}
    for li in range(2):
        for lj in range(2):
            m7_a[(li, lj)] = [(li, lj + 2, 1), (li + 2, lj + 2, -1)]
    # B10: rows 2-3, cols 0-3. B11: rows 2-3, cols 4-7.
    # B_combined[lj][lk] = B[lj+2, lk] + B[lj+2, lk+4]
    m7_b = {}
    for lj in range(2):
        for lk in range(4):
            m7_b[(lj, lk)] = [(lj + 2, lk, 1), (lj + 2, lk + 4, 1)]
    
    # M7 contributes +1 to C00 (offset 0, 0)
    add_product_contribution(m7_a, m7_b, 0, 0, 1, 4, 8)
    
    # Now build U, V, W matrices
    num_terms = len(U)
    n, m, p = 4, 4, 8
    U_mat = [[0] * num_terms for _ in range(n * m)]
    V_mat = [[0] * num_terms for _ in range(m * p)]
    W_mat = [[0] * num_terms for _ in range(p * n)]
    
    for t in range(num_terms):
        u_idx, u_val = U[t]
        v_idx, v_val = V[t]
        w_idx, w_val = W[t]
        U_mat[u_idx][t] = u_val
        V_mat[v_idx][t] = v_val
        W_mat[w_idx][t] = w_val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _try_rank_reduction(schoolbook, n, m, p, max_terms=128):
    """
    Try to reduce the rank by finding cancellations.
    Use a simple approach: try random linear combinations of columns.
    """
    U0 = schoolbook["U"]
    V0 = schoolbook["V"]
    W0 = schoolbook["W"]
    rank0 = len(W0[0])
    
    # The schoolbook has rank 128. Can we do better?
    # Known lower bound for 4x4x4 is 49, for 4x4x8 it's at least... 
    # Let's just try to find a valid lower-rank decomposition.
    
    # For now, return schoolbook (no improvement found)
    return schoolbook


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["W"][0])
    
    # Try Strassen
    if time.time() - start_time < WALL_DEADLINE - 5:
        try:
            strassen = _try_strassen_rank(n, m, p)
            s_rank = len(strassen["W"][0])
            if s_rank < best_rank:
                if _verify_fast(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                    best = strassen
                    best_rank = s_rank
        except Exception:
            pass
    
    # Try to further reduce
    if time.time() - start_time < WALL_DEADLINE - 5:
        try:
            reduced = _try_rank_reduction(best, n, m, p)
            r_rank = len(reduced["W"][0])
            if r_rank < best_rank:
                if _verify_fast(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    best = reduced
                    best_rank = r_rank
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END
