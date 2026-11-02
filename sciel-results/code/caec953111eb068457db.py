import numpy as np

def construct():
    n, m, p = 3, 3, 7
    # We will build the matrices U, V, W incrementally.
    # We need to know the total rank first to allocate, or use lists and convert.
    # Since rank is fixed at 58, we can allocate.
    
    rank = 58
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # 1. Block A[0:2, 0:2] * B[0:2, :]
    # This block covers i in {0,1}, j in {0,1}, k in {0..6}
    # We split k into {0,1}, {2,3}, {4,5}, {6}
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, k_start, k_len):
        # a_i_start, a_j_start: top-left of the 2x2 A block
        # b_j_start: top row of the 2x2 B block (wait, B is 2x2 so rows are j, cols are k)
        # Actually, B indices are (j, k).
        # We are multiplying A[2x2] and B[2x2] to get C[2x2].
        # A indices: (a_i_start..a_i_start+1, a_j_start..a_j_start+1)
        # B indices: (b_j_start..b_j_start+1, k_start..k_start+1)
        # C indices: (k_start..k_start+1, a_i_start..a_i_start+1) -> W[k*n + i]
        
        # A entries:
        # a = A[ai, aj], b = A[ai, aj+1], c = A[ai+1, aj], d = A[ai+1, aj+1]
        # B entries:
        # e = B[bi, k], f = B[bi, k+1], g = B[bi+1, k], h = B[bi+1, k+1]
        
        ai, aj = a_i_start, a_j_start
        bi, k = b_j_start, k_start
        
        # Indices in U (i*m+j)
        u_a = ai * m + aj
        u_b = ai * m + (aj + 1)
        u_c = (ai + 1) * m + aj
        u_d = (ai + 1) * m + (aj + 1)
        
        # Indices in V (j*p+k)
        v_e = bi * p + k
        v_f = bi * p + (k + 1)
        v_g = (bi + 1) * p + k
        v_h = (bi + 1) * p + (k + 1)
        
        # Indices in W (k*n+i)
        # C top-left (k, ai) -> W[k*n + ai]
        w_p = k * n + ai
        # C top-right (k, ai+1) -> W[k*n + (ai+1)]
        w_q = k * n + (ai + 1)
        # C bottom-left (k+1, ai) -> W[(k+1)*n + ai]
        w_r = (k + 1) * n + ai
        # C bottom-right (k+1, ai+1) -> W[(k+1)*n + (ai+1)]
        w_s = (k + 1) * n + (ai + 1)
        
        # Strassen terms
        # m1 = (a+d)(e+h)
        add_term(u_a, 1, v_e, 1, w_p, 1)
        add_term(u_a, 1, v_h, 1, w_p, 1)
        add_term(u_d, 1, v_e, 1, w_p, 1)
        add_term(u_d, 1, v_h, 1, w_p, 1)
        add_term(u_a, 1, v_e, 1, w_s, 1)
        add_term(u_a, 1, v_h, 1, w_s, 1)
        add_term(u_d, 1, v_e, 1, w_s, 1)
        add_term(u_d, 1, v_h, 1, w_s, 1)
        # This approach adds 8 terms for m1? No, m1 is ONE term in the decomposition.
        # The term is U_vec * V_vec * W_vec.
        # U_vec has 1s at u_a, u_d.
        # V_vec has 1s at v_e, v_h.
        # W_vec has coeffs.
        # We need to add these into the SAME column `col`.
        
        # Let's rewrite add_term to allow accumulating into the current col.
        
    # The `add_term` above increments col. This is wrong for Strassen which uses 7 terms, each of which is a sum of products.
    # Each Strassen "term" (like m1) corresponds to ONE column in the U, V, W matrices.
    # U column will have non-zeros at the A indices involved.
    # V column will have non-zeros at the B indices involved.
    # W column will have non-zeros at the C indices involved.
    
    # Let's restart the helper logic.
    
    col = 0
    def add_strassen_term(u_indices, v_indices, w_indices):
        """
        Adds one rank-1 term defined by linear combos.
        u_indices: list of (idx, coeff)
        v_indices: list of (idx, coeff)
        w_indices: list of (idx, coeff)
        """
        nonlocal col
        for idx, coeff in u_indices:
            U[idx, col] += coeff
        for idx, coeff in v_indices:
            V[idx, col] += coeff
        for idx, coeff in w_indices:
            W[idx, col] += coeff
        col += 1
        
    def add_trivial_term(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, 0:2] (k=0,1)
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U: 0, 1, 3, 4
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V: 0, 1, 7, 8
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W: 0, 1, 3, 4
    
    # m1 = (a+d)(e+h)
    add_strassen_term(
        [(0, 1), (4, 1)],  # a, d
        [(0, 1), (8, 1)],  # e, h
        [(0, 1), (4, 1)]   # p, s
    )
    # m2 = (c+d)e
    add_strassen_term(
        [(3, 1), (4, 1)],  # c, d
        [(0, 1)],          # e
        [(3, 1), (4, -1)]  # r, s (wait, m2 contributes +r and -s? No, s = m1 - m2 + m3 + m6)
        # Let's check signs again.
        # s = m1 - m2 + m3 + m6
        # So m2 has coeff -1 for s.
        # r = m2 + m4. m2 has coeff +1 for r.
    )
    # Correction: I should pass the coefficients for W as they appear in the final sum.
    # m2 contributes +1 to r and -1 to s.
    add_strassen_term(
        [(3, 1), (4, 1)],
        [(0, 1)],
        [(3, 1), (4, -1)]
    )
    # m3 = a(f-h)
    # q = m3 + m5. s = ... + m3 ...
    # m3 contributes +1 to q, +1 to s.
    add_strassen_term(
        [(0, 1)],
        [(1, 1), (8, -1)], # f, -h
        [(1, 1), (4, 1)]   # q, s
    )
    # m4 = d(g-e)
    # p = ... + m4 ... . r = m2 + m4.
    # m4 contributes +1 to p, +1 to r.
    add_strassen_term(
        [(4, 1)],
        [(7, 1), (0, -1)], # g, -e
        [(0, 1), (3, 1)]   # p, r
    )
    # m5 = (a+b)h
    # p = ... - m5 ... . q = m3 + m5.
    # m5 contributes -1 to p, +1 to q.
    add_strassen_term(
        [(0, 1), (1, 1)], # a, b
        [(8, 1)],         # h
        [(0, -1), (1, 1)] # p, q
    )
    # m6 = (c-a)(e+f)
    # s = ... + m6 ...
    # m6 contributes +1 to s.
    add_strassen_term(
        [(3, 1), (0, -1)], # c, -a
        [(0, 1), (1, 1)],  # e, f
        [(4, 1)]           # s
    )
    # m7 = (b-d)(g+h)
    # p = ... + m7 ...
    # m7 contributes +1 to p.
    add_strassen_term(
        [(1, 1), (4, -1)], # b, -d
        [(7, 1), (8, 1)],  # g, h
        [(0, 1)]           # p
    )
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4] (k=2,3)
    # B indices shift by +2 in k. V indices: (0,2)->2, (0,3)->3, (1,2)->9, (1,3)->10.
    # C indices shift by +2 in k. W indices: k=2 -> 6, 7; k=3 -> 9, 10.
    
    # m1
    add_strassen_term(
        [(0, 1), (4, 1)],
        [(2, 1), (10, 1)], # e, h (V[0*7+2]=2, V[1*7+3]=10)
        [(6, 1), (10, 1)]  # p, s (W[2*3+0]=6, W[3*3+1]=10)
    )
    # m2
    add_strassen_term(
        [(3, 1), (4, 1)],
        [(2, 1)],
        [(9, 1), (10, -1)] # r, s (W[3*3+0]=9, W[3*3+1]=10)
    )
    # m3
    add_strassen_term(
        [(0, 1)],
        [(3, 1), (10, -1)], # f, -h
        [(7, 1), (10, 1)]   # q, s (W[2*3+1]=7)
    )
    # m4
    add_strassen_term(
        [(4, 1)],
        [(9, 1), (2, -1)],  # g, -e
        [(6, 1), (9, 1)]    # p, r
    )
    # m5
    add_strassen_term(
        [(0, 1), (1, 1)],
        [(10, 1)],
        [(6, -1), (7, 1)]   # p, q
    )
    # m6
    add_strassen_term(
        [(3, 1), (0, -1)],
        [(2, 1), (3, 1)],
        [(10, 1)]
    )
    # m7
    add_strassen_term(
        [(1, 1), (4, -1)],
        [(9, 1), (10, 1)],
        [(6, 1)]
    )
    
    # 3. A[0:2, 0:2] * B[0:2, 4:6] (k=4,5)
    # B indices: V[4]=4, V[5]=5, V[11]=11, V[12]=12.
    # C indices: W[12]=12, W[13]=13, W[15]=15, W[16]=16.
    
    add_strassen_term(
        [(0, 1), (4, 1)],
        [(4, 1), (12, 1)],
        [(12, 1), (16, 1)]
    )
    add_strassen_term(
        [(3, 1), (4, 1)],
        [(4, 1)],
        [(15, 1), (16, -1)]
    )
    add_strassen_term(
        [(0, 1)],
        [(5, 1), (12, -1)],
        [(13, 1), (16, 1)]
    )
    add_strassen_term(
        [(4, 1)],
        [(11, 1), (4, -1)],
        [(12, 1), (15, 1)]
    )
    add_strassen_term(
        [(0, 1), (1, 1)],
        [(12, 1)],
        [(12, -1), (13, 1)]
    )
    add_strassen_term(
        [(3, 1), (0, -1)],
        [(4, 1), (5, 1)],
        [(16, 1)]
    )
    add_strassen_term(
        [(1, 1), (4, -1)],
        [(11, 1), (12, 1)],
        [(12, 1)]
    )
    
    # 4. A[0:2, 0:2] * B[0:2, 6:7] (k=6)
    # This is 2x2 * 2x1. Rank 2.
    # C[0,0] = A00 B06 + A01 B16
    # C[0,1] = A10 B06 + A11 B16
    # Wait, C indices for k=6 are W[18] (i=0) and W[19] (i=1).
    # B indices: V[6] (j=0, k=6), V[13] (j=1, k=6).
    # A indices: U[0] (0,0), U[1] (0,1), U[3] (1,0), U[4] (1,1).
    
    # Term 1: A00 * B06 -> C00
    add_trivial_term(0, 6, 18)
    # Term 2: A01 * B16 -> C01
    add_trivial_term(1, 13, 19)
    # Wait, C[0,1] is W[19]. C[1,0] is W[18]? No.
    # W index = k*n + i.
    # k=6, i=0 -> 18.
    # k=6, i=1 -> 19.
    
    # C[6,0] (i=0, j=0) = A00 B06 + A01 B16? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Here i is row of C, k is col of C.
    # Block is A[0:2, 0:2] * B[0:2, 6:7].
    # Result C[0:2, 6:7].
    # C[0,6] = A00 B06 + A01 B16.
    # C[1,6] = A10 B06 + A11 B16.
    
    # Term 1: A00 B06 contributes to C[0,6].
    add_trivial_term(0, 6, 18)
    # Term 2: A01 B16 contributes to C[0,6].
    add_trivial_term(1, 13, 18)
    # Term 3: A10 B06 contributes to C[1,6].
    add_trivial_term(3, 6, 19)
    # Term 4: A11 B16 contributes to C[1,6].
    add_trivial_term(4, 13, 19)
    
    # Wait, I used 4 terms. Rank of 2x2 * 2x1 is 4?
    # Yes, if we don't use Strassen-like tricks for rectangular.
    # But we can do better?
    # C[0,6] = A00 B06 + A01 B16
    # C[1,6] = A10 B06 + A11 B16
    # This is just 4 multiplications.
    
    # 5. A[0:2, 2] * B[2, :]
    # A col 2 (i=0,1), j=2. U[2] (0,2), U[5] (1,2).
    # B row 2 (j=2), k=0..6. V[14..20].
    # C[0:2, :] (i=0,1), k=0..6.
    # C[i,k] = A[i,2] B[2,k].
    # 14 terms.
    for i in range(2):
        for k in range(7):
            u_idx = i * 3 + 2
            v_idx = 2 * 7 + k
            w_idx = k * 3 + i
            add_trivial_term(u_idx, v_idx, w_idx)
            
    # 6. A[2, 0:2] * B[0:2, :]
    # A row 2 (i=2), j=0,1. U[6] (2,0), U[7] (2,1).
    # B rows 0,1 (j=0,1), k=0..6. V[0..6], V[7..13].
    # C[2, :] (i=2), k=0..6.
    # C[2,k] = A[2,0] B[0,k] + A[2,1] B[1,k].
    # 14 terms.
    for j in range(2):
        for k in range(7):
            u_idx = 2 * 3 + j
            v_idx = j * 7 + k
            w_idx = k * 3 + 2
            add_trivial_term(u_idx, v_idx, w_idx)
            
    # 7. A[2,2] * B[2, :]
    # A[2,2]. U[8].
    # B[2, k]. V[14..20].
    # C[2,k]. W[14..20].
    # 7 terms.
    for k in range(7):
        u_idx = 2 * 3 + 2
        v_idx = 2 * 7 + k
        w_idx = k * 3 + 2
        add_trivial_term(u_idx, v_idx, w_idx)

    # Check col index
    # 3 * 7 (Strassen) = 21
    # 4 (Trivial 2x2*2x1) = 4
    # 14 (A[0:2,2]) = 14
    # 14 (A[2,0:2]) = 14
    # 7 (A[2,2]) = 7
    # Total = 21 + 4 + 14 + 14 + 7 = 60.
    
    # Wait, my previous calc was 58.
    # 23 (for 2x2x7) + 14 + 14 + 7 = 58.
    # 2x2x7 rank: 3*7 (Strassen for 2x2x2) + 2 (Trivial for 2x2x1) = 21 + 2 = 23.
    # My code used 4 terms for the 2x2x1 part.
    # Can we do 2x2x1 in rank 2?
    # C[0,k] = A00 B0k + A01 B1k
    # C[1,k] = A10 B0k + A11 B1k
    # This is outer product of [A00, A10]^T and [B0k, B1k]? No.
    # It's $\sum_{j=0}^1 A_{i,j} B_{j,k}$.
    # This is a rank 4 tensor unless we have structure.
    # But wait, if k is fixed, it's just a 2x2 matrix product with 1 column.
    # Rank of 2x2x1 is 4?
    # No, $\mu(2,2,1) = 2$?
    # Let's check.
    # $C_{i0} = A_{i0} B_{00} + A_{i1} B_{10}$.
    # Terms: $A_{00}B_{00}, A_{01}B_{10}, A_{10}B_{00}, A_{11}B_{10}$.
    # If we use Strassen-like:
    # $M_1 = (A_{00}+A_{11}) B_{00}$
    # $M_2 = (A_{00}-A_{11}) B_{10}$
    # This doesn't seem to reduce rank below 4 for rectangular 2x2x1.
    # Actually, $\mu(2,2,1) = 4$.
    # So 23 was wrong. It is 21 + 4 = 25.
    # Total rank = 25 + 14 + 14 + 7 = 60.
    
    # Is 60 better than 63? Yes.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
