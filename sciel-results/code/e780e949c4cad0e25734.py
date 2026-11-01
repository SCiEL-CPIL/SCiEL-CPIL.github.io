"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Kronecker product decomposition exploiting 6 = 3*2.
    
    Decompose T_{2,3,6} = T_{1,1,2} ⊗ T_{2,3,3}
    
    Check: n1*n2=1*2=2, m1*m2=1*3=3, p1*p2=2*3=6. Correct!
    
    T_{1,1,2}: rank 2 (scalar times 1x2 vector)
    T_{2,3,3}: rank 18 (schoolbook for 2x3*3x3)
    
    Total rank = 2 * 18 = 36. Same as schoolbook, but let's implement it.
    
    Actually, let me try T_{2,3,2} ⊗ T_{1,1,3}:
    n: 2*1=2, m: 3*1=3, p: 2*3=6. Correct!
    Rank = 12 * 3 = 36.
    
    Both give rank 36. Let me just implement the Kronecker product properly
    to verify correctness, even though rank is the same.
    """
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 1x1 * 1x2 (rank 2)
    n1, m1, p1 = 1, 1, 2
    rank1 = n1 * m1 * p1  # = 2
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    
    # Sub-problem 2: 2x3 * 3x3 (rank 18)
    n2, m2, p2 = 2, 3, 3
    rank2 = n2 * m2 * p2  # = 18
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    # Combine via Kronecker product
    # T1 ⊗ T2 gives T_{n1*n2, m1*m2, p1*p2}
    # Index mapping for U: (i1*m1+j1, i2*m2+j2) -> (i1*n2*m2 + i2*m2 + j2) ... 
    # Actually, the combined U has shape (n*m, rank1*rank2)
    # Row index in combined: ((i1*m1+j1) * (n2*m2) + (i2*m2+j2))
    # But we need it to match i*m+j where i = i1*n2+i2, j = j1*m2+j2
    # i*m+j = (i1*n2+i2)*m + j1*m2+j2 where m=m1*m2
    # = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    # Row in U1 is i1*m1+j1, row in U2 is i2*m2+j2
    # Combined row = (i1*m1+j1)*(n2*m2) + (i2*m2+j2)
    # = i1*m1*n2*m2 + j1*n2*m2 + i2*m2 + j2
    # This should equal i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    # These are NOT the same in general!
    
    # I need to be more careful with the index mapping.
    # The Kronecker product tensor T1⊗T2 has components:
    # T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # where i = i1*n2+i2, j = j1*m2+j2, k = k1*p2+k2
    # 
    # For the decomposition, each rank-1 term is indexed by (r1, r2) where
    # r1 is a rank-1 index for T1 and r2 for T2.
    # U_combined[i, r1, r2] = U1[i1, r1] * U2[i2, r2] where i = i1*n2+i2
    # But i1 and i2 are determined by i: i1 = i // n2, i2 = i % n2
    # Similarly j1 = j // m2, j2 = j % m2
    # And k1 = k // p2, k2 = k % p2
    
    rank_total = rank1 * rank2
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            # For each (i1, j1, k1) contributing to U1[i1*m1+j1, r1]
            for i1 in range(n1):
                for j1 in range(m1):
                    if U1[i1 * m1 + j1][r1] != 0:
                        for i2 in range(n2):
                            for j2 in range(m2):
                                if U2[i2 * m2 + j2][r2] != 0:
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    U[i * m + j][r] += U1[i1 * m1 + j1][r1] * U2[i2 * m2 + j2][r2]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    if V1[j1 * p1 + k1][r1] != 0:
                        for j2 in range(m2):
                            for k2 in range(p2):
                                if V2[j2 * p2 + k2][r2] != 0:
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    V[j * p + k][r] += V1[j1 * p1 + k1][r1] * V2[j2 * p2 + k2][r2]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    if W1[k1 * n1 + i1][r1] != 0:
                        for k2 in range(p2):
                            for i2 in range(n2):
                                if W2[k2 * n2 + i2][r2] != 0:
                                    k = k1 * p2 + k2
                                    i = i1 * n2 + i2
                                    W[k * n + i][r] += W1[k1 * n1 + i1][r1] * W2[k2 * n2 + i2][r2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
