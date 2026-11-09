"""Rewritten program for 4x4 * 4x9 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    """
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    return U, V, W

def strassen_recursive(n, m, p):
    """
    Returns U, V, W for n x m x p Strassen decomposition.
    n, m, p must be powers of 2.
    """
    if n == 2 and m == 2 and p == 2:
        return strassen_2x2()
    
    n2, m2, p2 = n // 2, m // 2, p // 2
    U_sub, V_sub, W_sub = strassen_recursive(n2, m2, p2)
    R_sub = len(U_sub[0])
    
    U = [[0] * (7 * R_sub) for _ in range(n * m)]
    V = [[0] * (7 * R_sub) for _ in range(m * p)]
    W = [[0] * (7 * R_sub) for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a list of (coeff, block_index) for A and B
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ([(1, (0,0)), (1, (1,1))], [(1, (0,0)), (1, (1,1))]),
        # M2 = (A10 + A11)B00
        ([(1, (1,0)), (1, (1,1))], [(1, (0,0))]),
        # M3 = A00(B01 - B11)
        ([(1, (0,0))], [(1, (0,1)), (-1, (1,1))]),
        # M4 = A11(B10 - B00)
        ([(1, (1,1))], [(1, (1,0)), (-1, (0,0))]),
        # M5 = (A00 + A01)B11
        ([(1, (0,0)), (1, (0,1))], [(1, (1,1))]),
        # M6 = (A10 - A00)(B00 + B01)
        ([(1, (1,0)), (-1, (0,0))], [(1, (0,0)), (1, (0,1))]),
        # M7 = (A01 - A11)(B10 + B11)
        ([(1, (0,1)), (-1, (1,1))], [(1, (1,0)), (1, (1,1))])
    ]
    
    for s, (a_coeffs, b_coeffs) in enumerate(terms):
        for r in range(R_sub):
            R_global = s * R_sub + r
            
            # Accumulate U, V, W contributions
            # We need to sum over all pairs of A-blocks and B-blocks
            for c_a, (k, l) in a_coeffs:
                for c_b, (l_prime, m_prime) in b_coeffs:
                    c_total = c_a * c_b
                    if c_total == 0:
                        continue
                    
                    # Update U
                    for i_prime in range(n2):
                        for j_prime in range(m2):
                            idx_sub = i_prime * m2 + j_prime
                            val = U_sub[idx_sub][r]
                            if val != 0:
                                idx_global = (k * n2 + i_prime) * m + (l * m2 + j_prime)
                                U[idx_global][R_global] += c_total * val
                                
                    # Update V
                    for j_prime in range(m2):
                        for k_prime in range(p2):
                            idx_sub = j_prime * p2 + k_prime
                            val = V_sub[idx_sub][r]
                            if val != 0:
                                idx_global = (l_prime * m2 + j_prime) * p + (m_prime * p2 + k_prime)
                                V[idx_global][R_global] += c_total * val
                                
                    # Update W
                    for k_prime in range(p2):
                        for i_prime in range(n2):
                            idx_sub = k_prime * n2 + i_prime
                            val = W_sub[idx_sub][r]
                            if val != 0:
                                idx_global = (m_prime * p2 + k_prime) * n + (k * n2 + i_prime)
                                W[idx_global][R_global] += c_total * val
                                
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    U1, V1, W1 = strassen_recursive(4, 4, 4)
    R1 = len(U1[0])
    
    # Part 2: A (4x4) * B2 (4x4) -> C2 (4x4)
    # B2 corresponds to columns 4-7 of B
    U2, V2, W2 = strassen_recursive(4, 4, 4)
    R2 = len(U2[0])
    
    # Part 3: A (4x4) * B3 (4x1) -> C3 (4x1)
    # B3 corresponds to column 8 of B
    # Schoolbook rank 16
    U3 = [[0] * 16 for _ in range(16)]
    V3 = [[0] * 16 for _ in range(4)] # m*p = 4*1 = 4
    W3 = [[0] * 16 for _ in range(4)] # p*n = 1*4 = 4
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U3[i * 4 + j][r] = 1
            V3[j * 1 + 0][r] = 1
            W3[0 * 4 + i][r] = 1
            
    R3 = 16
    
    Total_R = R1 + R2 + R3
    
    U = [[0] * Total_R for _ in range(16)]
    V = [[0] * Total_R for _ in range(36)]
    W = [[0] * Total_R for _ in range(36)]
    
    # Map Part 1
    # B1 cols 0-3. k in 0..3
    for r in range(R1):
        for i in range(4):
            for j in range(4):
                I = i * 4 + j
                U[I][r] += U1[I][r]
                # V index in sub: j * 4 + k_sub. Global: j * 9 + k_sub
                for k_sub in range(4):
                    J_sub = j * 4 + k_sub
                    val = V1[J_sub][r]
                    if val != 0:
                        J_global = j * 9 + k_sub
                        V[J_global][r] += val
                # W index in sub: k_sub * 4 + i. Global: k_sub * 4 + i
                for k_sub in range(4):
                    K_sub = k_sub * 4 + i
                    val = W1[K_sub][r]
                    if val != 0:
                        K_global = k_sub * 4 + i
                        W[K_global][r] += val
                        
    # Map Part 2
    # B2 cols 4-7. k in 4..7. k_sub in 0..3, k_global = 4 + k_sub
    offset2 = R1
    for r in range(R2):
        for i in range(4):
            for j in range(4):
                I = i * 4 + j
                U[I][offset2 + r] += U2[I][r]
                for k_sub in range(4):
                    J_sub = j * 4 + k_sub
                    val = V2[J_sub][r]
                    if val != 0:
                        J_global = j * 9 + (4 + k_sub)
                        V[J_global][offset2 + r] += val
                for k_sub in range(4):
                    K_sub = k_sub * 4 + i
                    val = W2[K_sub][r]
                    if val != 0:
                        K_global = (4 + k_sub) * 4 + i
                        W[K_global][offset2 + r] += val
                        
    # Map Part 3
    # B3 col 8. k = 8.
    offset3 = R1 + R2
    for r in range(R3):
        for i in range(4):
            for j in range(4):
                I = i * 4 + j
                U[I][offset3 + r] += U3[I][r]
                # V3 index: j * 1 + 0 = j. Global: j * 9 + 8
                val = V3[j][r]
                if val != 0:
                    J_global = j * 9 + 8
                    V[J_global][offset3 + r] += val
                # W3 index: 0 * 4 + i = i. Global: 8 * 4 + i = 32 + i
                val = W3[i][r]
                if val != 0:
                    K_global = 32 + i
                    W[K_global][offset3 + r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
