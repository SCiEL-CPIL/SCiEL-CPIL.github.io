"""Rewritten program using Kronecker-inspired factorization for 2x3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Decompose M_{2,3,5} as M_{2,1,5} (Kronecker) M_{1,3,1}
    # M_1: n1=2, m1=1, p1=5, Rank R1 = 10
    # M_2: n2=1, m2=3, p2=1, Rank R2 = 3
    # Total Rank R = R1 * R2 = 30
    
    n1, m1, p1 = 2, 1, 5
    n2, m2, p2 = 1, 3, 1
    R1 = n1 * m1 * p1  # 10
    R2 = n2 * m2 * p2  # 3
    R = R1 * R2        # 30
    
    # Initialize U, V, W for total tensor
    # U shape: (n*m, R) = (6, 30)
    # V shape: (m*p, R) = (15, 30)
    # W shape: (p*n, R) = (10, 30)
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Construct M_1 decomposition (U1, V1, W1)
    # U1 shape (n1*m1, R1) = (2, 10)
    # V1 shape (m1*p1, R1) = (1, 10)
    # W1 shape (p1*n1, R1) = (5, 10)
    U1 = [[0] * R1 for _ in range(n1 * m1)]
    V1 = [[0] * R1 for _ in range(m1 * p1)]
    W1 = [[0] * R1 for _ in range(p1 * n1)]
    
    r1 = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1][r1] = 1
                V1[j1 * p1 + k1][r1] = 1
                W1[k1 * n1 + i1][r1] = 1
                r1 += 1
                
    # Construct M_2 decomposition (U2, V2, W2)
    # U2 shape (n2*m2, R2) = (3, 3)
    # V2 shape (m2*p2, R2) = (3, 3)
    # W2 shape (p2*n2, R2) = (1, 3)
    U2 = [[0] * R2 for _ in range(n2 * m2)]
    V2 = [[0] * R2 for _ in range(m2 * p2)]
    W2 = [[0] * R2 for _ in range(p2 * n2)]
    
    r2 = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2 * m2 + j2][r2] = 1
                V2[j2 * p2 + k2][r2] = 1
                W2[k2 * n2 + i2][r2] = 1
                r2 += 1
                
    # Combine via Kronecker Product
    # Index mapping:
    # i_total = i1 * n2 + i2
    # j_total = j1 * m2 + j2
    # k_total = k1 * p2 + k2
    # r_total = r1 * R2 + r2
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                # Find r1 corresponding to (i1, j1, k1) in M1
                # In schoolbook, r1 = i1*m1*p1 + j1*p1 + k1
                r1_idx = i1 * m1 * p1 + j1 * p1 + k1
                
                # Iterate over M2 indices to fill the Kronecker block
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # Find r2 corresponding to (i2, j2, k2) in M2
                            r2_idx = i2 * m2 * p2 + j2 * p2 + k2
                            
                            r_total = r1_idx * R2 + r2_idx
                            
                            # Calculate total indices
                            i_tot = i1 * n2 + i2
                            j_tot = j1 * m2 + j2
                            k_tot = k1 * p2 + k2
                            
                            # U index: i_tot * m + j_tot
                            u_idx = i_tot * m + j_tot
                            # V index: j_tot * p + k_tot
                            v_idx = j_tot * p + k_tot
                            # W index: k_tot * n + i_tot
                            w_idx = k_tot * n + i_tot
                            
                            # Values from sub-decompositions
                            val_u = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                            val_v = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                            val_w = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                            
                            U[u_idx][r_total] += val_u
                            V[v_idx][r_total] += val_v
                            W[w_idx][r_total] += val_w
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
