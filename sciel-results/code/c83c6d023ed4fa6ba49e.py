import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    rank = 55
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)
    # W: (p*n, rank) = (20, 55)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x10 block (Rank 20)
    # Corresponds to j=2 (0-indexed) in the middle dimension
    # A indices: i*3 + 2
    # B indices: 2*10 + k = 20 + k
    # C indices: k*2 + i
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 20
    
    # Part 2: 2x2x10 block (Rank 35)
    # Corresponds to j in {0, 1}
    # Split k into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair q, use Strassen rank 7
    
    # Strassen coefficients for U (A matrix)
    # A is 2x2: a00, a01, a10, a11
    # U indices: i*3 + j
    # a00 -> U[0*3+0] = U[0]
    # a01 -> U[0*3+1] = U[1]
    # a10 -> U[1*3+0] = U[3]
    # a11 -> U[1*3+1] = U[4]
    
    # Strassen coefficients for V (B matrix)
    # B is 2x2: b00, b01, b10, b11
    # V indices: j*10 + k
    # b00 -> V[0*10 + 2q]
    # b01 -> V[0*10 + 2q+1]
    # b10 -> V[1*10 + 2q]
    # b11 -> V[1*10 + 2q+1]
    
    # Strassen coefficients for W (C matrix)
    # C is 2x2: c00, c01, c10, c11
    # W indices: k*2 + i
    # c00 -> W[2q*2 + 0] = W[4q]
    # c01 -> W[(2q+1)*2 + 0] = W[4q+2]
    # c10 -> W[2q*2 + 1] = W[4q+1]
    # c11 -> W[(2q+1)*2 + 1] = W[4q+3]
    
    # Define the 7 Strassen steps
    # Each step is a tuple: (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # We will generate these dynamically for each q
    
    for q in range(5):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # V indices for this block
        v_b00 = 0 * p + k0
        v_b01 = 0 * p + k1
        v_b10 = 1 * p + k0
        v_b11 = 1 * p + k1
        
        # W indices for this block
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        # U indices (same for all q)
        u_a00 = 0 * m + 0
        u_a01 = 0 * m + 1
        u_a10 = 1 * m + 0
        u_a11 = 1 * m + 1
        
        # Step 0: M1 = (a00 + a11)(b00 + b11)
        c = col
        U[u_a00][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
        W[w_c11][c] = 1
        col += 1
        
        # Step 1: M2 = (a10 + a11)b00
        c = col
        U[u_a10][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1
        W[w_c11][c] = -1
        col += 1
        
        # Step 2: M3 = a00(b01 - b11)
        c = col
        U[u_a00][c] = 1
        V[v_b01][c] = 1
        V[v_b11][c] = -1
        W[w_c01][c] = 1
        W[w_c11][c] = 1
        col += 1
        
        # Step 3: M4 = a11(b10 - b00)
        c = col
        U[u_a11][c] = 1
        V[v_b10][c] = 1
        V[v_b00][c] = -1
        W[w_c00][c] = 1
        W[w_c10][c] = 1
        col += 1
        
        # Step 4: M5 = (a00 + a01)b11
        c = col
        U[u_a00][c] = 1
        U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1
        W[w_c01][c] = 1
        col += 1
        
        # Step 5: M6 = (a10 - a00)(b00 + b01)
        c = col
        U[u_a10][c] = 1
        U[u_a00][c] = -1
        V[v_b00][c] = 1
        V[v_b01][c] = 1
        W[w_c11][c] = 1
        col += 1
        
        # Step 6: M7 = (a01 - a11)(b10 + b11)
        c = col
        U[u_a01][c] = 1
        U[u_a11][c] = -1
        V[v_b10][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
