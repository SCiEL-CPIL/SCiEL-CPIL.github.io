def _block_decomposition(n, m, p):
    """Build rank-24 decomposition by 4 blocks of 6 terms each."""
    n, m, p = 2, 3, 8
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 6
        
        # Split i into two 1x3x2 blocks
        for i in range(n):
            # Rank 3 for 1x3x2
            # T_jk = 1 = sum_r V_jr W_kr
            # We need 3 terms to handle the summation over j?
            # Actually, for 1x3x2, the rank is 3.
            # Decomposition:
            # r=0: V=[1,0,0], W=[1,1] -> A0(B0+B1)
            # r=1: V=[0,1,0], W=[1,1] -> A1(B0+B1)
            # r=2: V=[0,0,1], W=[1,1] -> A2(B0+B1)
            # Sum: (A0+A1+A2)(B0+B1).
            # This computes C_k = (A0+A1+A2)(B0+B1) for k=0,1.
            # But we want C_k = A0 B0k + A1 B1k + A2 B2k.
            # The above computes (A0+A1+A2)B0k + (A0+A1+A2)B1k.
            # This is NOT correct.
            
            # We need T_jk = 1.
            # C_k = sum_j A_j B_jk.
            # If we use rank 3:
            # r=0: V=[1,0,0], W=[1,0] -> A0 B00
            # r=1: V=[0,1,0], W=[1,0] -> A1 B10
            # r=2: V=[0,0,1], W=[1,0] -> A2 B20
            # This only computes C0.
            # We need C1 too.
            # So we need more terms.
            
            # Rank of 1x3x2 is actually 3?
            # Let's check rank of 1x3x2 tensor.
            # It is the sum of 3 rank-1 tensors?
            # T_jk = 1.
            # 1 = sum_r V_jr W_kr.
            # If we use 3 terms:
            # r=0: V=[1,0,0], W=[1,0] -> 1 at (0,0)
            # r=1: V=[0,1,0], W=[1,0] -> 1 at (1,0)
            # r=2: V=[0,0,1], W=[1,0] -> 1 at (2,0)
            # Sum is 1 at (j,0). 0 at (j,1).
            # This is not all ones.
            
            # To get all ones, we need:
            # sum_r V_jr W_kr = 1 for all j,k.
            # This requires the matrix M_jk to be all ones.
            # Rank of M is 1.
            # So 1 term is enough for the tensor.
            # V=[1,1,1], W=[1,1].
            # Then C_k = sum_j A_j B_jk * 1 * 1?
            # No.
            # The bilinear form is sum_j A_j B_jk.
            # If we use 1 term, we compute M = sum_j A_j B_jk?
            # M is a vector.
            # So we cannot use 1 term.
            
            # The rank of the tensor 1x3x2 is 3.
            # Why?
            # Because the bilinear map (A, B) -> C is not separable into 1 product.
            # It requires 3 products?
            # C0 = A0B00 + A1B10 + A2B20
            # C1 = A0B01 + A1B11 + A2B21
            # We can compute:
            # M0 = A0(B00+B01)
            # M1 = A1(B10+B11)
            # M2 = A2(B20+B21)
            # Then C0 = M0 + M1 + M2 - (A0B01 + A1B11 + A2B21)? No.
            
            # Actually, rank of 1x3x2 is 3.
            # Decomposition:
            # r=0: V=[1,0,0], W=[1,1] -> A0(B00+B01)
            # r=1: V=[0,1,0], W=[1,1] -> A1(B10+B11)
            # r=2: V=[0,0,1], W=[1,1] -> A2(B20+B21)
            # Sum: (A0+A1+A2)(B00+B01) etc? No.
            # C_k = sum_r W_kr M_r.
            # M_r = sum_j V_jr A_j B_jk?
            # If V=[1,0,0], M0 = A0 B0k.
            # Then C_k = 1*A0B0k + 1*A1B1k + 1*A2B2k.
            # This is correct.
            # So we need 3 terms.
            
            for r in range(3):
                idx = col + i * 3 + r
                # U
                U[i * m + r][idx] = 1 # A_ir
                # V
                V[r * p + k0][idx] = 1 # B_{r, k0}
                V[r * p + k0 + 1][idx] = 1 # B_{r, k0+1}
                # W
                W[k0 * n + i][idx] = 1
                W[(k0 + 1) * n + i][idx] = 1

    return {"U": U, "V": V, "W": W}
