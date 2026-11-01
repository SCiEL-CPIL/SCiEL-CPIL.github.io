"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank 35 decomposition using 5 independent Strassen blocks
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for a 2x2 block
    # A indices: 00, 01, 10, 11 -> U rows: 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> V rows: j*p + k
    # C indices: 00, 01, 10, 11 -> W rows: k*n + i
    
    # Define the 7 Strassen multipliers for one block
    # Each multiplier is a tuple (U_vec, V_vec, W_vec)
    # U_vec is dict {row: coeff}
    # V_vec is dict {row: coeff}
    # W_vec is dict {row: coeff}
    
    # Block b corresponds to B columns 2*b and 2*b+1
    # V row for B[0, 2*b] is 0*p + 2*b = 2*b
    # V row for B[0, 2*b+1] is 0*p + 2*b+1 = 2*b+1
    # V row for B[1, 2*b] is 1*p + 2*b = 10 + 2*b
    # V row for B[1, 2*b+1] is 1*p + 2*b+1 = 10 + 2*b+1
    
    # W row for C[0, 2*b] is 2*b*n + 0 = 4*b
    # W row for C[0, 2*b+1] is (2*b+1)*n + 0 = 4*b+2
    # W row for C[1, 2*b] is 2*b*n + 1 = 4*b+1
    # W row for C[1, 2*b+1] is (2*b+1)*n + 1 = 4*b+3
    
    # We iterate over 5 blocks
    for b in range(5):
        base_col = b * 7
        # V indices for this block
        v00 = 0 * p + 2 * b
        v01 = 0 * p + 2 * b + 1
        v10 = 1 * p + 2 * b
        v11 = 1 * p + 2 * b + 1
        
        # W indices for this block
        w00 = 2 * b * n + 0
        w01 = (2 * b + 1) * n + 0
        w10 = 2 * b * n + 1
        w11 = (2 * b + 1) * n + 1
        
        # U indices (same for all blocks)
        u00 = 0
        u01 = 1
        u10 = 2
        u11 = 3
        
        # Multiplier 1: (A00+A11)(B00+B11)
        U[u00][base_col + 0] = 1
        U[u11][base_col + 0] = 1
        V[v00][base_col + 0] = 1
        V[v11][base_col + 0] = 1
        W[w00][base_col + 0] = 1
        W[w11][base_col + 0] = 1
        
        # Multiplier 2: (A10+A11)B00
        U[u10][base_col + 1] = 1
        U[u11][base_col + 1] = 1
        V[v00][base_col + 1] = 1
        W[w10][base_col + 1] = 1
        W[w11][base_col + 1] = -1
        
        # Multiplier 3: A00(B01-B11)
        U[u00][base_col + 2] = 1
        V[v01][base_col + 2] = 1
        V[v11][base_col + 2] = -1
        W[w01][base_col + 2] = 1
        W[w11][base_col + 2] = 1
        
        # Multiplier 4: A11(B10-B00)
        U[u11][base_col + 3] = 1
        V[v10][base_col + 3] = 1
        V[v00][base_col + 3] = -1
        W[w00][base_col + 3] = 1
        W[w10][base_col + 3] = 1
        
        # Multiplier 5: (A00+A01)B11
        U[u00][base_col + 4] = 1
        U[u01][base_col + 4] = 1
        V[v11][base_col + 4] = 1
        W[w00][base_col + 4] = -1
        W[w01][base_col + 4] = 1
        
        # Multiplier 6: (A10-A00)(B00+B01)
        U[u10][base_col + 5] = 1
        U[u00][base_col + 5] = -1
        V[v00][base_col + 5] = 1
        V[v01][base_col + 5] = 1
        W[w11][base_col + 5] = 1
        
        # Multiplier 7: (A01-A11)(B10+B11)
        U[u01][base_col + 6] = 1
        U[u11][base_col + 6] = -1
        V[v10][base_col + 6] = 1
        V[v11][base_col + 6] = 1
        W[w00][base_col + 6] = 1
        W[w01][base_col + 6] = -1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
