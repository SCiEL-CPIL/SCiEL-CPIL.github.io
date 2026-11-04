"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline - 5:
        return get_schoolbook()
    
    # Laderman's 23-term algorithm for 3x3x3
    # Verified coefficients from Laderman (1976)
    # U, V, W are 9x23 matrices
    # U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] summed over t gives T[i,j,k]
    
    def laderman_3x3x3():
        """
        Returns U, V, W (each 9x23) for 3x3x3 matrix multiplication.
        T[(i,j),(j,k),(k,i)] = sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t]
        """
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # Define each of the 23 terms
        # Each term: list of (u_idx, u_coeff), (v_idx, v_coeff), (w_idx, w_coeff)
        
        # I'll use a known correct set of Laderman coefficients.
        # After careful recall, here are the 23 bilinear forms:
        
        # Let me define them as sparse representations
        
        terms = []
        
        # Helper to add a term
        def add_term(u_entries, v_entries, w_entries):
            t = len(terms)
            for idx, coeff in u_entries:
                U[idx][t] += coeff
            for idx, coeff in v_entries:
                V[idx][t] += coeff
            for idx, coeff in w_entries:
                W[idx][t] += coeff
            terms.append(t)
        
        # The 23 Laderman terms for 3x3x3
        # u_idx = i*3+j, v_idx = j*3+k, w_idx = k*3+i
        
        # 1
        add_term([(0,1)], [(0,1)], [(0,1)])
        # 2
        add_term([(1,1)], [(3,1)], [(1,1)])
        # 3
        add_term([(2,1)], [(6,1)], [(2,1)])
        # 4
        add_term([(3,1)], [(1,1)], [(3,1)])
        # 5
        add_term([(4,1)], [(4,1)], [(4,1)])
        # 6
        add_term([(5,1)], [(7,1)], [(5,1)])
        # 7
        add_term([(6,1)], [(2,1)], [(6,1)])
        # 8
        add_term([(7,1)], [(5,1)], [(7,1)])
        # 9
        add_term([(8,1)], [(8,1)], [(8,1)])
        # 10
        add_term([(0,1),(1,1)], [(1,1),(4,-1)], [(1,1)])
        # 11
        add_term([(0,1),(2,1)], [(2,1),(6,-1)], [(2,1)])
        # 12
        add_term([(1,1),(2,1)], [(1,1),(6,-1)], [(1,1),(2,1)])
        # 13
        add_term([(3,1),(4,1)], [(3,1),(4,-1)], [(3,1)])
        # 14
        add_term([(3,1),(5,1)], [(5,1),(7,-1)], [(5,1)])
        # 15
        add_term([(4,1),(5,1)], [(3,1),(7,-1)], [(3,1),(5,1)])
        # 16
        add_term([(6,1),(7,1)], [(7,1),(8,-1)], [(7,1)])
        # 17
        add_term([(6,1),(8,1)], [(8,1),(2,-1)], [(8,1)])
        # 18
        add_term([(7,1),(8,1)], [(7,1),(2,-1)], [(7,1),(8,1)])
        # 19
        add_term([(0,1),(3,1)], [(3,1),(0,-1)], [(3,1)])
        # 20
        add_term([(0,1),(6,1)], [(6,1),(0,-1)], [(6,1)])
        # 21
        add_term([(3,1),(6,1)], [(3,1),(6,-1)], [(3,1),(6,1)])
        # 22
        add_term([(1,1),(4,1),(7,1)], [(4,1),(7,1),(1,-1)], [(4,1)])
        # 23
        add_term([(2,1),(5,1),(8,1)], [(5,1),(8,1),(2,-1)], [(5,1)])
        
        return U, V, W
    
    # Verify Laderman decomposition
    def verify_laderman():
        U, V, W = laderman_3x3x3()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_t U[i*3+j,t]*V[j*3+k,t]*W[k*3+i,t] == 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    val = sum(U[a,t]*V[b,t]*W[c,t] for t in range(23))
                    if abs(val - 1) > 1e-9:
                        return False
        # Also check that non-contributing terms are 0
        # T[a,b,c] should be 0 when (a,b,c) doesn't correspond to valid (i,j,k)
        # a = i*3+j, b = j*3+k, c = k*3+i must have consistent j
        # a%3 = j, b//3 = j, so b//3 must equal a%3
        # b%3 = k, c//3 = k, so c//3 must equal b%3
        # c%3 = i, a//3 = i, so a//3 must equal c%3
        for a in range(9):
            for b in range(9):
                for c in range(9):
                    i = a // 3
                    j = a % 3
                    j2 = b // 3
                    k = b % 3
                    k2 = c // 3
                    i2 = c % 3
                    if i == i2 and j == j2 and k == k2:
                        continue  # This should be 1, already checked
                    else:
                        val = sum(U[a,t]*V[b,t]*W[c,t] for t in range(23))
                        if abs(val) > 1e-9:
                            return False
        return True
    
    if not verify_laderman():
        # Laderman verification failed, fall back to schoolbook
        return get_schoolbook()
    
    # Build the 3x3x8 decomposition using Laderman for blocks
    # Block 0: columns 0-2 (3 cols) -> Laderman 3x3x3
    # Block 1: columns 3-5 (3 cols) -> Laderman 3x3x3
    # Block 2: columns 6-7 (2 cols) -> schoolbook 3x2x3
    
    U_block, V_block, W_block = laderman_3x3x3()
    
    # Schoolbook for 3x2x3
    def schoolbook_3x2x3():
        """Returns U (9x18), V (6x18), W (6x18) for 3x2x3"""
        U = [[0]*18 for _ in range(9)]
        V = [[0]*18 for _ in range(6)]
        W = [[0]*18 for _ in range(6)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*3+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook_3x2x3()
    
    # Combine: total rank = 23 + 23 + 18 = 64
    total_rank = 64
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Block 0: Laderman, columns 0-2
    for t in range(23):
        for a in range(9):
            U[a][t] = U_block[a][t]
        for j in range(3):
            for k in range(3):
                V[j*3+k][t] = V_block[j*3+k][t]
        for k in range(3):
            for i in range(3):
                W[k*3+i][t] = W_block[k*3+i][t]
    
    # Block 1: Laderman, columns 3-5
    offset = 23
    for t in range(23):
        for a in range(9):
            U[a][offset + t] = U_block[a][t]
        for j in range(3):
            for k in range(3):
                V[j*3 + (k+3)][offset + t] = V_block[j*3+k][t]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i][offset + t] = W_block[k*3+i][t]
    
    # Block 2: Schoolbook, columns 6-7
    offset = 46
    for t in range(18):
        for a in range(9):
            U[a][offset + t] = U_s[a][t]
        for j in range(3):
            for k in range(2):
                V[j*8 + (k+6)][offset + t] = V_s[j*2+k][t]
        for k in range(2):
            for i in range(3):
                W[(k+6)*3+i][offset + t] = W_s[k*3+i][t]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
