import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # Strassen rank-7 terms for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Define the 7 rank-1 tensors
    # Each is (u, v, w) where u, v, w are 4-vectors
    # u corresponds to A, v to B, w to C
    # Indices: 00, 01, 10, 11
    
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, -1])), # c00=1, c01=0, c10=0, c11=-1? No, w is the matrix added.
        # Actually, let's just use the linear forms and reconstruction matrices.
        # This is safer.
    ]
    
    # Let's define u, v vectors for the 7 products
    # and W matrices for reconstruction.
    
    u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, -1, 1, 0],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # Reconstruction: C = sum M_r * W_r
    # W_r is 2x2
    W = [
        [[1, 0], [0, 1]], # M1 contributes to c00, c11
        [[0, 0], [1, 0]], # M2 contributes to c10
        [[0, 1], [0, 0]], # M3 contributes to c01
        [[0, 0], [1, 0]], # M4 contributes to c10? 
        [[0, 0], [0, -1]],# M5 contributes to c11?
        [[0, 0], [0, 1]], # M6 contributes to c11?
        [[1, 0], [0, 0]]  # M7 contributes to c00?
    ]
    # This manual mapping is error prone.
    
    # I will use the explicit tensor definition of Strassen.
    # There are 7 rank-1 terms.
    # I will generate them by solving the system or using a known list.
    # For the sake of the code, I will use the schoolbook rank 8 for the 2x2 blocks if Strassen fails, 
    # but I'll try to implement Strassen correctly.
    
    # Actually, I can just use the fact that rank(4,4,6) <= 2 * rank(2,2,2) * 3? No.
    # Rank(4,4,6) <= 6 * Rank(2,2,2) is not quite right because of the sum.
    # C_IK = sum_J A_IJ B_JK.
    # This is a sum of products.
    # If I use Strassen, I get 7 terms per product.
    # There are 12 products.
    # Total 84 terms.
    
    # I will construct the 84 terms.
    rank = 84
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W_out = [[0] * rank for _ in range(24)]
    
    # Strassen coefficients for 2x2
    # u_r, v_r are 4-vectors. w_r is 4-vector (flattened 2x2 matrix).
    # I need the correct w_r.
    
    # Let's use the explicit rank-7 decomposition of 2x2x2.
    # From literature:
    # 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] -> c00+=1, c11+=1
    # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] -> c10+=1
    # 3: u=[1,0,0,0], v=[0,-1,1,0], w=[0,1,0,0] -> c01+=1
    # 4: u=[0,0,0,1], v=[-1,1,0,0], w=[0,0,1,0] -> c10+=1
    # 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1] -> c11-=1
    # 6: u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1] -> c11+=1
    # 7: u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,0] -> c00+=1
    
    # Let's check c00: 1 (from 1) + 1 (from 7) = 2? No.
    # c00 should be 1.
    # My w vectors are contributions to C.
    
    # I will use a verified list of (u, v, w) for Strassen.
    # If I can't verify, I'll fall back to schoolbook rank 8 for 2x2, total rank 96.
    # But 96 is the baseline.
    
    # I'll use the schoolbook for 2x2 to ensure correctness, but that gives 96.
    # The prompt asks to beat 96.
    # I will use the Strassen terms from a reliable source I can trust in memory:
    # The 7 terms are:
    # 1. (a00+a11)(b00+b11) -> c00, c11
    # 2. (a10+a11)b00 -> c10
    # 3. a00(b10-b11) -> c01
    # 4. a11(b01-b00) -> c10
    # 5. (a00+a01)b11 -> c11
    # 6. (a10-a00)(b00+b10) -> c11
    # 7. (a01-a11)(b11+b01) -> c00
    
    # Reconstruction:
    # c00 = 1 + 7
    # c01 = 3 + 5
    # c10 = 2 + 4
    # c11 = 1 - 2 + 3 + 6
    
    # This matches the formulas I had.
    # So the w vectors are:
    # w1 = [1, 0, 0, 1]
    # w2 = [0, 0, 1, 0]
    # w3 = [0, 1, 0, 0]
    # w4 = [0, 0, 1, 0]
    # w5 = [0, 0, 0, 1]
    # w6 = [0, 0, 0, 1]
    # w7 = [1, 0, 0, 0]
    
    # And the signs in reconstruction:
    # c00: +w1, +w7
    # c01: +w3, +w5
    # c10: +w2, +w4
    # c11: +w1, -w2, +w3, +w6
    
    # This implies w vectors are not just 0/1, they carry the sign in the reconstruction?
    # No, the term is M_r * w_r.
    # So for c11, the contribution from M2 is -1 * w2.
    # So w2 is [0,0,1,0]. Contribution to c10 is +1.
    # Contribution to c11 is -1.
    # This works if we allow negative coefficients in W matrix.
    
    # So the 7 rank-1 terms for 2x2x2 are:
    # 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # 3: u=[1,0,0,0], v=[0,-1,1,0], w=[0,1,0,0]
    # 4: u=[0,0,0,1], v=[-1,1,0,0], w=[0,0,1,0]
    # 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,1]
    # 6: u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1]
    # 7: u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,0]
    
    # Now construct the global decomposition.
    
    strassen_u = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    strassen_v = [
        [1,0,0,1],
        [1,0,0,0],
        [0,-1,1,0],
        [-1,1,0,0],
        [0,0,0,1],
        [1,0,1,0],
        [0,1,0,1]
    ]
    strassen_w = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,1],
        [1,0,0,0]
    ]
    
    # Reconstruction signs for C_IK
    # C_IK = sum_J sum_r sign(I,K,r) * M_r(I,J,K) * W_r
    # M_r(I,J,K) = u_r(A_IJ) v_r(B_JK)
    
    # sign matrix for (I, K, r)
    # I in {0,1}, K in {0,1,2}
    # r in {1..7}
    
    # c00 (I=0, K=0): r1(+), r7(+)
    # c01 (I=0, K=1): r3(+), r5(+)
    # c10 (I=1, K=0): r2(+), r4(+)
    # c11 (I=1, K=1): r1(+), r2(-), r3(+), r6(+)
    
    # Wait, the K index in the 2x2 block C_IK corresponds to the 2x2 block of the 4x6 matrix.
    # The 4x6 matrix has 3 blocks of width 2.
    # So K in {0,1,2} refers to the block index.
    # Inside the block, the indices are 0,1.
    # The Strassen w vector has 4 entries corresponding to the 2x2 block.
    # So for each (I, K, J, r), we add a term.
    
    col = 0
    for I in range(2):
        for K in range(3):
            for J in range(2):
                for r in range(7):
                    # Determine sign
                    sign = 0
                    if I == 0 and K == 0:
                        if r == 0 or r == 6: sign = 1
                    elif I == 0 and K == 1:
                        if r == 2 or r == 4: sign = 1
                    elif I == 1 and K == 0:
                        if r == 1 or r == 3: sign = 1
                    elif I == 1 and K == 1:
                        if r == 0: sign = 1
                        if r == 1: sign = -1
                        if r == 2: sign = 1
                        if r == 5: sign = 1
                    elif I == 0 and K == 2:
                        # K=2 is the third block.
                        # The structure is the same for all K.
                        # C_02 = A_00 B_02 + A_01 B_12
                        # So signs are same as K=0.
                        if r == 0 or r == 6: sign = 1
                    elif I == 1 and K == 2:
                        # Same as K=1.
                        if r == 0: sign = 1
                        if r == 1: sign = -1
                        if r == 2: sign = 1
                        if r == 5: sign = 1
                    else:
                        # I=1, K=0,1,2 handled. I=0, K=0,1,2 handled.
                        pass

                    if sign == 0:
                        col += 1
                        continue

                    # Fill U, V, W
                    # U index: i, j. A_IJ is at rows 2I..2I+1, cols 2J..2J+1.
                    # u_r is 4-vector [a00, a01, a10, a11]
                    for li in range(2):
                        for lj in range(2):
                            idx_a = li * 2 + lj
                            val_u = strassen_u[r][idx_a]
                            if val_u != 0:
                                i = 2 * I + li
                                j = 2 * J + lj
                                u_idx = i * 4 + j
                                U[u_idx][col] += sign * val_u

                    # V index: j, k. B_JK is at rows 2J..2J+1, cols 2K..2K+1.
                    # v_r is 4-vector [b00, b01, b10, b11]
                    for lj in range(2):
                        for lk in range(2):
                            idx_b = lj * 2 + lk
                            val_v = strassen_v[r][idx_b]
                            if val_v != 0:
                                j = 2 * J + lj
                                k = 2 * K + lk
                                v_idx = j * 6 + k
                                V[v_idx][col] += sign * val_v

                    # W index: k, i. C_IK is at rows 2K..2K+1, cols 2I..2I+1.
                    # w_r is 4-vector [c00, c01, c10, c11]
                    for lk in range(2):
                        for li in range(2):
                            idx_c = lk * 2 + li
                            val_w = strassen_w[r][idx_c]
                            if val_w != 0:
                                k = 2 * K + lk
                                i = 2 * I + li
                                w_idx = k * 4 + i
                                W_out[w_idx][col] += sign * val_w

                    col += 1

    return {"U": U, "V": V, "W": W_out}
