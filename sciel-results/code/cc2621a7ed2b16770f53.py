import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Try rank 54 decomposition
    # 3 blocks of 3 cols (rank 16 each) + 1 block of 1 col (rank 6)
    rank_54 = 54
    U = np.zeros((n * m, rank_54), dtype=float)
    V = np.zeros((m * p, rank_54), dtype=float)
    W = np.zeros((p * n, rank_54), dtype=float)
    
    col_idx = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col_idx
        if col_idx >= rank_54:
            return False
        U[u_idx, col_idx] += val
        V[v_idx, col_idx] += val
        W[w_idx, col_idx] += val
        col_idx += 1
        return True

    # Helper to add M_2,1,3 terms (rank 6)
    # A is 2x1 (a0, a1), B is 1x3 (b0, b1, b2)
    # C is 2x3
    def add_m213(a0, a1, b0, b1, b2, c00, c01, c02, c10, c11, c12):
        nonlocal col_idx
        if not add_term(a0, b0, c00): return False
        if not add_term(a0, b1, c01): return False
        if not add_term(a0, b2, c02): return False
        if not add_term(a1, b0, c10): return False
        if not add_term(a1, b1, c11): return False
        if not add_term(a1, b2, c12): return False
        return True

    # Helper to add M_2,2,3 terms (rank 10)
    # A is 2x2 (a00, a01, a10, a11), B is 2x3 (b00, b01, b02, b10, b11, b12)
    # C is 2x3
    # This is the tricky part. I will use a known rank 10 decomposition.
    # If I don't have it, I will use rank 11 (7+4) and fall back to 55 if total > 55.
    # Actually, let's just use rank 11 for M_2,2,3 to be safe, which gives rank 57.
    # But 57 is worse than 55.
    # So I MUST use rank 10 or stick to 55.
    # I will stick to 55 if I can't implement rank 10 easily.
    
    # Let's try to implement rank 10 for M_2,2,3.
    # One known decomposition for M_2,2,3 (rank 10):
    # It is not trivial to write from memory.
    # I will use the rank 55 solution as the primary, and only try 54 if I can verify it.
    
    # Since I cannot guarantee the rank 10 coefficients, I will return the rank 55 solution.
    # The rank 55 solution is robust.
    
    # Re-implementing rank 55 solution
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col_idx = 0
    
    def add_strassen_term(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col_idx
        if time.time() > deadline:
            return False
        # M1
        U[a00, col_idx] += 1.0; U[a11, col_idx] += 1.0
        V[b00, col_idx] += 1.0; V[b11, col_idx] += 1.0
        W[c00, col_idx] += 1.0; W[c11, col_idx] += 1.0
        col_idx += 1
        # M2
        U[a10, col_idx] += 1.0; U[a11, col_idx] += 1.0
        V[b00, col_idx] += 1.0
        W[c10, col_idx] += 1.0; W[c11, col_idx] -= 1.0
        col_idx += 1
        # M3
        U[a00, col_idx] += 1.0
        V[b01, col_idx] += 1.0; V[b11, col_idx] -= 1.0
        W[c01, col_idx] += 1.0; W[c11, col_idx] += 1.0
        col_idx += 1
        # M4
        U[a11, col_idx] += 1.0
        V[b10, col_idx] += 1.0; V[b00, col_idx] -= 1.0
        W[c00, col_idx] += 1.0; W[c10, col_idx] += 1.0
        col_idx += 1
        # M5
        U[a00, col_idx] += 1.0; U[a01, col_idx] += 1.0
        V[b11, col_idx] += 1.0
        W[c00, col_idx] -= 1.0; W[c01, col_idx] += 1.0
        col_idx += 1
        # M6
        U[a10, col_idx] += 1.0; U[a00, col_idx] -= 1.0
        V[b00, col_idx] += 1.0; V[b01, col_idx] += 1.0
        W[c11, col_idx] += 1.0
        col_idx += 1
        # M7
        U[a01, col_idx] += 1.0; U[a11, col_idx] -= 1.0
        V[b10, col_idx] += 1.0; V[b11, col_idx] += 1.0
        W[c00, col_idx] += 1.0
        col_idx += 1
        return True

    def add_schoolbook_term(a_i, b_j, c_ik):
        nonlocal col_idx
        if time.time() > deadline:
            return False
        U[a_i, col_idx] += 1.0
        V[b_j, col_idx] += 1.0
        W[c_ik, col_idx] += 1.0
        col_idx += 1
        return True

    for b_idx in range(5):
        k0 = 2 * b_idx
        k1 = 2 * b_idx + 1
        a00 = 0; a01 = 1; a10 = 3; a11 = 4; a02 = 2; a12 = 5
        b00 = 0 * p + k0; b01 = 0 * p + k1
        b10 = 1 * p + k0; b11 = 1 * p + k1
        b20 = 2 * p + k0; b21 = 2 * p + k1
        c00 = k0 * n + 0; c01 = k1 * n + 0
        c10 = k0 * n + 1; c11 = k1 * n + 1
        
        if not add_strassen_term(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
            break
        if not add_schoolbook_term(a02, b20, c00): break
        if not add_schoolbook_term(a02, b21, c01): break
        if not add_schoolbook_term(a12, b20, c10): break
        if not add_schoolbook_term(a12, b21, c11): break

    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
