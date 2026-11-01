"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
    # Also verify that non-matching entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                a = i * m + j
                                b = jp * p + kp
                                c = k * n + ip
                                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                                if val != 0:
                                    return False
    return True

def _rank11_2x3_3x2():
    """
    Rank-11 bilinear algorithm for 2x3 * 3x2.
    Convention: U[i*3+j, d], V[j*2+k, d], W[k*2+i, d]
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]
    C = [[c00, c01],
         [c10, c11]]
    
    c[i,k] = sum_j a[i,j] * b[j,k]
    
    Known rank-11 construction:
    t0  = a00 * b00
    t1  = a00 * b01
    t2  = a01 * b10
    t3  = a01 * b11
    t4  = a02 * b20
    t5  = a02 * b21
    t6  = (a00+a10) * b00
    t7  = (a00+a10) * b01
    t8  = (a01+a11) * b10
    t9  = (a01+a11) * b11
    t10 = (a02+a12) * (b20+b21)
    
    Then:
    c00 = t0 + t2 + t4
    c01 = t1 + t3 + t5
    c10 = (t6-t0) + (t8-t2) + (t10 - t5 - (t10 - t5))  -- need to fix
    
    Actually, let me use a verified construction.
    """
    n, m, p = 2, 3, 2
    r = 11
    U = [[0] * r for _ in range(n * m)]  # 6 x 11
    V = [[0] * r for _ in range(m * p)]  # 6 x 11
    W = [[0] * r for _ in range(p * n)]  # 4 x 11
    
    # Verified rank-11 algorithm:
    # Use the following 11 bilinear terms:
    # 
    # Term 0:  a00 * b00       -> contributes to c00
    # Term 1:  a00 * b01       -> contributes to c01
    # Term 2:  a01 * b10       -> contributes to c00
    # Term 3:  a01 * b11       -> contributes to c01
    # Term 4:  a02 * b20       -> contributes to c00
    # Term 5:  a02 * b21       -> contributes to c01
    # Term 6:  (a00+a10) * b00 -> contributes to c00, c10
    # Term 7:  (a00+a10) * b01 -> contributes to c01, c11
    # Term 8:  (a01+a11) * b10 -> contributes to c00, c10
    # Term 9:  (a01+a11) * b11 -> contributes to c01, c11
    # Term 10: (a02+a12) * (b20+b21) -> contributes to c00,c01,c10,c11
    
    # From these:
    # c00 = t0 + t2 + t4
    # c01 = t1 + t3 + t5
    # c10 = (t6-t0) + (t8-t2) + [from t10: a12*b20 + a12*b21 - a02*b20 - a02*b21]
    #     = (t6-t0) + (t8-t2) + (t10 - t4 - t5)
    # c11 = (t7-t1) + (t9-t3) + (t10 - t4 - t5)
    #     Wait, that gives c11 = (t7-t1) + (t9-t3) + (t10 - t4 - t5)
    #     And c10 = (t6-t0) + (t8-t2) + (t10 - t4 - t5)
    #     But c10 should have a12*b20 and c11 should have a12*b21
    #     t10 = (a02+a12)*(b20+b21) = a02*b20 + a02*b21 + a12*b20 + a12*b21
    #     So t10 - t4 - t5 = a12*b20 + a12*b21
    #     This contributes to BOTH c10 and c11, which is wrong.
    
    # Let me try a different approach. Use a known valid rank-11.
    # Based on literature, here's a valid construction:
    
    # I'll set up the matrices and verify programmatically.
    # For now, let me use a construction that I can verify:
    
    # Terms:
    # 0: a00*b00
    # 1: a00*b01  
    # 2: a01*b10
    # 3: a01*b11
    # 4: a02*b20
    # 5: a02*b21
    # 6: (a00+a10)*b00
    # 7: (a00+a10)*b01
    # 8: (a01+a11)*b10
    # 9: (a01+a11)*b11
    # 10: (a02+a12)*b20
    
    # Then:
    # c00 = t0 + t2 + t4
    # c01 = t1 + t3 + t5
    # c10 = (t6-t0) + (t8-t2) + (t10-t4)
    # c11 = (t7-t1) + (t9-t3) + [missing a12*b21]
    
    # Still missing! Need a12*b21.
    
    # Alternative: swap which terms get the "combined" treatment.
    # Use:
    # 0: a00*b00
    # 1: a00*b01
    # 2: a01*b10
    # 3: a01*b11
    # 4: a02*b20
    # 5: a02*b21
    # 6: (a00+a10)*b00
    # 7: (a00+a10)*b01
    # 8: (a01+a11)*b10
    # 9: (a01+a11)*b11
    # 10: (a02+a12)*b21
    
    # Then:
    # c00 = t0 + t2 + t4
    # c01 = t1 + t3 + t5
    # c10 = (t6-t0) + (t8-t2) + [missing a12*b20]
    # c11 = (t7-t1) + (t9-t3) + (t10-t5)
    
    # Still missing one!
    
    # The fundamental issue: with 11 terms, we have 6 "base" terms for i=0
    # and 5 "combined" terms. We need to cover all 12 products.
    # 6 base + 5 combined = 11, but we need 12 distinct products.
    # Each combined term covers 2 products, so 6 + 5*2 = 16 > 12.
    # But the overlaps must be exact.
    
    # Let me just use rank 12 (schoolbook) for the sub-block and tile.
    r = 12
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Use a direct construction with the sharing trick.
    # For 2x3 * 3x8:
    # 
    # Key insight: for each output column k, we need:
    #   c[0,k] = sum_j a[0,j]*b[j,k]
    #   c[1,k] = sum_j a[1,j]*b[j,k]
    #
    # Use the trick: for each j, define
    #   s_j = a[0,j]
    #   t_j = a[0,j] + a[1,j]
    # Then a[1,j] = t_j - s_j
    #
    # For each (j,k):
    #   Product 1: a[0,j] * b[j,k]  (for c[0,k])
    #   Product 2: (a[0,j]+a[1,j]) * b[j,k]  (for c[0,k] and c[1,k])
    #   Then a[1,j]*b[j,k] = Product2 - Product1
    #
    # This gives 2*3*8 = 48 terms. No savings.
    #
    # To save, we need to share across k. For a fixed j, instead of 8 separate
    # products with b[j,0],...,b[j,7], we can use linear combinations.
    #
    # For fixed j, define 8 products:
    #   p_k = a[0,j] * b[j,k] for k=0..7
    #   q_k = (a[0,j]+a[1,j]) * b[j,k] for k=0..7
    # Total: 16 per j, 48 total.
    #
    # Alternative: use 8 products per j by combining:
    #   For k in 0..7:
    #     r_k = a[0,j] * b[j,k]
    #     s_k = a[1,j] * b[j,k]
    #   c[0,k] += r_k, c[1,k] += s_k
    # Total: 16 per j, 48 total. Same.
    #
    # The minimum rank for 2x3*3x8 is at most 48.
    # Known bounds: rank >= 32 (from various lower bound techniques).
    # 
    # Let me try a tiling approach with a better sub-algorithm.
    # If I can find a rank-11 for 2x3*3x2, tiling 4 times gives rank 44.
    #
    # For now, return schoolbook rank 48.
    
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    assert _verify(U, V, W, n, m, p, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
