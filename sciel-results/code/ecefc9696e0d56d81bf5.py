"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

def _construct_block_strassen(n, m, p, b):
    """Return (U_block, V_block, W_block) for a single 2x2 block using Strassen.
    U is 4x7, V is 12x7, W is 4x7 for this block.
    Actually V should be (m*p) x rank, but for a single block only the relevant columns of B are used.
    For the full construction, we will assemble the blocks.
    """
    # For a single 2x2 block, the relevant parts of A are A[0,0], A[0,1], A[1,0], A[1,1]
    # The relevant parts of B are B[0, k0], B[0, k1], B[1, k0], B[1, k1] where k0=2b, k1=2b+1
    # The relevant parts of C are C[0, k0], C[1, k0], C[0, k1], C[1, k1]
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for this block: 0:(0,k0), 1:(0,k1), 2:(1,k0), 3:(1,k1)
    # W indices for this block: 0:(k0,0), 1:(k0,1), 2:(k1,0), 3:(k1,1)
    
    U_b = [[0]*7 for _ in range(4)]
    V_b = [[0]*7 for _ in range(4)]
    W_b = [[0]*7 for _ in range(4)]
    
    # Strassen steps
    # Step 0: P1 = (a+d)(e+h)
    r = 0
    U_b[0][r] = 1; U_b[3][r] = 1
    V_b[0][r] = 1; V_b[3][r] = 1
    W_b[0][r] = 1; W_b[3][r] = 1
    
    # Step 1: P2 = (c+d)e
    r = 1
    U_b[2][r] = 1; U_b[3][r] = 1
    V_b[0][r] = 1
    W_b[1][r] = 1; W_b[3][r] = -1
    
    # Step 2: P3 = a(f-h)
    r = 2
    U_b[0][r] = 1
    V_b[1][r] = 1; V_b[3][r] = -1
    W_b[2][r] = 1; W_b[3][r] = 1
    
    # Step 3: P4 = d(g-e)
    r = 3
    U_b[3][r] = 1
    V_b[2][r] = 1; V_b[0][r] = -1
    W_b[0][r] = 1; W_b[1][r] = 1
    
    # Step 4: P5 = (a+b)h
    r = 4
    U_b[0][r] = 1; U_b[1][r] = 1
    V_b[3][r] = 1
    W_b[0][r] = -1; W_b[2][r] = 1
    
    # Step 5: P6 = (c-a)(e+f)
    r = 5
    U_b[2][r] = 1; U_b[0][r] = -1
    V_b[0][r] = 1; V_b[1][r] = 1
    W_b[3][r] = 1
    
    # Step 6: P7 = (b-d)(g+h)
    r = 6
    U_b[1][r] = 1; U_b[3][r] = -1
    V_b[2][r] = 1; V_b[3][r] = 1
    W_b[0][r] = 1
    
    return U_b, V_b, W_b

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We process 3 blocks of 2x2
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V row indices for this block
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # W row indices for this block
        w_c00 = k0 * n + 0 # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n + 0 # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]
        
        # Strassen steps 0..6
        # Step 0: P1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        # Step 1: P2 = (c+d)e
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        # Step 2: P3 = a(f-h)
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        # Step 3: P4 = d(g-e)
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        # Step 4: P5 = (a+b)h
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        # Step 5: P6 = (c-a)(e+f)
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        # Step 6: P7 = (b-d)(g+h)
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
