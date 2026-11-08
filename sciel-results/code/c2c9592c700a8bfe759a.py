"""Direct construction of rank-65 tensor decomposition for 4x4 by 4x5 matrix multiplication."""

import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT_SECONDS = 110  # safety margin within the 120s evaluator limit
START_TIME = time.time()


def _elapsed() -> float:
    return time.time() - START_TIME


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False, "U dimension mismatch"
    if len(V) != m * p:
        return False, "V dimension mismatch"
    if len(W) != p * n:
        return False, "W dimension mismatch"
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False, "Rank mismatch"
    
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k] * 1 (for matrix mult)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta[j,?] ... 
    # The tensor for matmul is T[i,j,k] such that sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # In trace convention W[k*n+i, r], so W index is (k, i)
    # U[i*m + j, r], V[j*p + k, r]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the identity for matmul
    # Actually the matmul tensor T[i,j,k] = 1 (all ones) doesn't work.
    # The correct tensor: for each (i,j,k), the coefficient is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    # Wait no. The tensor decomposition satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k B[j,k]*V[j*p+k,r]) * ...
    # This is getting confusing. Let me just check numerically.
    
    # The matmul tensor T has T[i,j,k] = 1 for all valid i,j,k
    # The decomposition should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Spot check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False, f"T[{i},{j},{k}] = {s}, expected 1.0"
    return True, "OK"


def construct() -> dict:
    n, m, p = DIMENSIONS

    # Total rank = 49 (Strassen 4x4x4) + 16 (schoolbook 4x4x1)
    rank = 65

    U = [[0] * rank for _ in range(n * m)]  # (16, 65)
    V = [[0] * rank for _ in range(m * p)]  # (20, 65)
    W = [[0] * rank for _ in range(p * n)]  # (20, 65)

    col = 0

    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1

    # --- Strassen (4x4 * 4x4) via 7 outer x 7 inner terms ---

    # Outer (super) Strassen terms for 4x4 product:
    # A blocks: 0:A00, 1:A01, 2:A10, 3:A11
    # B blocks: 0:B00, 1:B01, 2:B10, 3:B11
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1: (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2: (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3: A00(B01-B11)
        ([0, 0, 0, 1], [-1, 0, 1, 0]),  # M4: A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),   # M5: (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0]),  # M6: (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1]),  # M7: (A01-A11)(B10+B11)
    ]

    # Contributions of M_t to C blocks (C00, C01, C10, C11)
    c_contrib = [
        [1, 0, 0, 1],    # M1
        [0, 0, 1, -1],   # M2
        [0, 1, 0, 1],    # M3
        [1, 0, 1, 0],    # M4
        [-1, 1, 0, 0],   # M5
        [0, 0, 0, 1],    # M6
        [1, 0, 0, 0],    # M7
    ]

    # Inner Strassen terms for a 2x2 product P*Q:
    # P elements: 0:P00, 1:P01, 2:P10, 3:P11
    # Q elements: 0:Q00, 1:Q01, 2:Q10, 3:Q11
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),        # s0 = (P00+P11)(Q00+Q11)
        ([(2, 1), (3, 1)], [(0, 1)]),                 # s1 = (P10+P11)Q00
        ([(0, 1)], [(1, 1), (3, -1)]),                # s2 = P00(Q01-Q11)
        ([(3, 1)], [(2, 1), (0, -1)]),                # s3 = P11(Q10-Q00)
        ([(0, 1), (1, 1)], [(3, 1)]),                 # s4 = (P00+P01)Q11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),        # s5 = (P10-P00)(Q00+Q01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),        # s6 = (P01-P11)(Q10+Q11)
    ]

    # Contributions of inner terms to S = P*Q (S00, S01, S10, S11)
    s_contrib = [
        [1, 0, 0, 1],   # s0
        [0, 0, 1, -1],  # s1
        [0, 1, 0, 1],   # s2
        [1, 0, 1, 0],   # s3
        [-1, 1, 0, 0],  # s4
        [0, 0, 0, 1],   # s5
        [1, 0, 0, 0],   # s6
    ]

    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    # Build the 49 Strassen terms
    for s_idx in range(7):
        if _elapsed() > TIME_LIMIT_SECONDS - 2:
            break
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            if _elapsed() > TIME_LIMIT_SECONDS - 2:
                break
            p_elems, q_elems = inner_terms[t_idx]

            u_list = []
            v_list = []
            w_list = []

            # Build U list
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))

            # Build V list
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))

            # Build W list
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0:
                        continue
                    total_coeff = sc * cc
                    if total_coeff == 0:
                        continue
                    if c_pos == 0:
                        gi, gk = s_r, s_c
                    elif c_pos == 1:
                        gi, gk = s_r, 2 + s_c
                    elif c_pos == 2:
                        gi, gk = 2 + s_r, s_c
                    elif c_pos == 3:
                        gi, gk = 2 + s_r, 2 + s_c
                    w_list.append((gk, gi, total_coeff))

            add_term(u_list, v_list, w_list)

    # --- Part 2: 4x4 * 4x1 using schoolbook (16 terms) ---
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)

    # Verify the decomposition (fast for rank 65, 400 entries * 65 = 26000 ops)
    ok, msg = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        # Fallback: pure schoolbook rank = n*m*p = 80
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
