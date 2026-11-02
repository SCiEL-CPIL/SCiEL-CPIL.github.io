"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: A is 2x2, B is 2x2, C is 2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        U = [
            [1, 0, 0, 0, 0, 0, 0], # A00
            [0, 1, 0, 0, 0, 0, 0], # A01
            [0, 0, 1, 0, 0, 0, 0], # A10
            [0, 0, 0, 1, 0, 0, 0], # A11
        ]
        V = [
            [1, 0, 0, 0, 0, 0, 0], # B00
            [0, 1, 0, 0, 0, 0, 0], # B01
            [0, 0, 1, 0, 0, 0, 0], # B10
            [0, 0, 0, 1, 0, 0, 0], # B11
        ]
        W = [
            [1, 0, 0, 0, 0, 0, 0], # C00
            [0, 1, 0, 0, 0, 0, 0], # C01
            [0, 0, 1, 0, 0, 0, 0], # C10
            [0, 0, 0, 1, 0, 0, 0], # C11
        ]
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U_strassen = [
            [1, 0, 1, 0, 1, 0, 0], # A00 in M1, M3, M5
            [0, 1, 0, 0, 1, 0, 0], # A01 in M2, M5
            [0, 0, 1, 0, 0, 1, 0], # A10 in M2, M6
            [1, 1, 0, 1, 0, 0, 0], # A11 in M1, M2, M4
        ]
        # Wait, the U matrix maps A indices to rank indices.
        # Rank index 0: M1 = (A00+A11)(B00+B11) -> A00:1, A11:1
        # Rank index 1: M2 = (A10+A11)B00 -> A10:1, A11:1
        # Rank index 2: M3 = A00(B01-B11) -> A00:1
        # Rank index 3: M4 = A11(B10-B00) -> A11:1
        # Rank index 4: M5 = (A00+A01)B11 -> A00:1, A01:1
        # Rank index 5: M6 = (A10-A00)(B00+B01) -> A10:1, A00:-1
        # Rank index 6: M7 = (A01-A11)(B10+B11) -> A01:1, A11:-1
        
        U_s = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        V_s = [
            [1, 1, 0, 0, 0, 1, 0],  # B00
            [0, 0, 1, 0, 0, 1, 1],  # B01
            [0, 0, 0, 1, 0, 0, 1],  # B10
            [1, 0, -1, -1, 1, 0, 1] # B11
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1], # C00 = M1 + M4 - M5 + M7
            [0, 0, 1, 0, 1, 0, 0],  # C01 = M3 + M5
            [0, 1, 0, 1, 0, 0, 0],  # C10 = M2 + M4
            [1, -1, 1, 0, 0, 1, 0]  # C11 = M1 - M2 + M3 + M6
        ]
        return U_s, V_s, W_s

    def get_schoolbook(n, m, p):
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    def combine_decomps(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
        # Combines two decompositions for blocks that sum to n, m, p
        # This function is a placeholder for the logic to be implemented in the main construct
        pass

    # Recursive construction
    def decompose(n, m, p):
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            return get_strassen_2x2x2()
            
        # Try to split m
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            # Split A into n x m1 and n x m2
            # Split B into m1 x p and m2 x p
            # C = A1 B1 + A2 B2
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            
            # Concatenate U, V, W
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U_new = [[0] * total_rank for _ in range(n*m)]
            V_new = [[0] * total_rank for _ in range(m*p)]
            W_new = [[0] * total_rank for _ in range(p*n)]
            
            # Fill U1
            for i in range(n):
                for j in range(m1):
                    for r in range(rank1):
                        U_new[i*m + j][r] = U1[i*m1 + j][r]
                        
            # Fill U2
            for i in range(n):
                for j in range(m2):
                    for r in range(rank2):
                        U_new[i*m + (m1 + j)][rank1 + r] = U2[i*m2 + j][r]
                        
            # Fill V1
            for j in range(m1):
                for k in range(p):
                    for r in range(rank1):
                        V_new[j*p + k][r] = V1[j*p + k][r]
                        
            # Fill V2
            for j in range(m2):
                for k in range(p):
                    for r in range(rank2):
                        V_new[(m1 + j)*p + k][rank1 + r] = V2[j*p + k][r]
                        
            # Fill W
            for k in range(p):
                for i in range(n):
                    for r in range(rank1):
                        W_new[k*n + i][r] = W1[k*n + i][r]
                    for r in range(rank2):
                        W_new[k*n + i][rank1 + r] = W2[k*n + i][r]
                        
            return U_new, V_new, W_new
            
        # Fallback to schoolbook if split not possible or not beneficial
        return get_schoolbook(n, m, p)

    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
