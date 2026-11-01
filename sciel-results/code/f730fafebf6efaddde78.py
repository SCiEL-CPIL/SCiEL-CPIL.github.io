import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # We aim for rank 32.
    # Structure: 4 blocks of 2 columns (Strassen rank 7 each) + 1 block of 1 column (Rank 4).
    # Total rank: 4 * 7 + 4 = 32.
    
    rank = 32
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, t):
        if u_val != 0:
            U[u_idx][t] += u_val
        if v_val != 0:
            V[v_idx][t] += v_val
        if w_val != 0:
            W[w_idx][t] += w_val
            
    # Define Strassen terms for a 2x2 block
    # Input A indices: 0,1,2,3
    # Input B indices (local): 0,1,2,3 (mapped to global)
    # Output C indices (local): 0,1,2,3 (mapped to global)
    
    # Strassen multipliers M1..M7
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We will instantiate 4 such blocks for k=0..1, 2..3, 4..5, 6..7
    
    for block in range(4):
        k_start = block * 2
        # Global V indices for B rows 0 and 1, cols k_start, k_start+1
        # B is 2x9. Index j*9 + k.
        v_e = k_start       # B[0, k0]
        v_f = k_start + 1   # B[0, k1]
        v_g = 9 + k_start   # B[1, k0]
        v_h = 9 + k_start + 1 # B[1, k1]
        
        # Global W indices for C rows 0 and 1, cols k_start, k_start+1
        # C is 9x2. Index k*2 + i.
        w_c00 = 2 * k_start       # C[0, k0]
        w_c01 = 2 * k_start + 1   # C[1, k0]
        w_c10 = 2 * (k_start + 1) # C[0, k1]
        w_c11 = 2 * (k_start + 1) + 1 # C[1, k1]
        
        base_t = block * 7
        
        # Term 1: M1
        t = base_t
        # U: a(0), d(3)
        add_term(0, 1, -1, 0, -1, 0, t) # Placeholder, will use specific logic below
        # Actually, let's just fill the arrays directly for speed and clarity in this specific case
        
        # Re-initializing the block contribution manually for precision
        
        # M1 = (a+d)(e+h) -> C00 +1, C11 +1
        t = base_t
        U[0][t] = 1; U[3][t] = 1
        V[v_e][t] = 1; V[v_h][t] = 1
        W[w_c00][t] = 1; W[w_c11][t] = 1
        
        # M2 = (c+d)e -> C10 +1, C11 -1
        t = base_t + 1
        U[2][t] = 1; U[3][t] = 1
        V[v_e][t] = 1
        W[w_c01][t] = 1; W[w_c11][t] = -1
        
        # M3 = a(f-h) -> C01 +1, C11 +1
        t = base_t + 2
        U[0][t] = 1
        V[v_f][t] = 1; V[v_h][t] = -1
        W[w_c10][t] = 1; W[w_c11][t] = 1
        
        # M4 = d(g-e) -> C00 +1, C10 +1
        t = base_t + 3
        U[3][t] = 1
        V[v_g][t] = 1; V[v_e][t] = -1
        W[w_c00][t] = 1; W[w_c01][t] = 1
        
        # M5 = (a+b)h -> C00 -1, C01 +1
        t = base_t + 4
        U[0][t] = 1; U[1][t] = 1
        V[v_h][t] = 1
        W[w_c00][t] = -1; W[w_c10][t] = 1
        
        # M6 = (c-a)(e+f) -> C11 +1
        t = base_t + 5
        U[2][t] = 1; U[0][t] = -1
        V[v_e][t] = 1; V[v_f][t] = 1
        W[w_c11][t] = 1
        
        # M7 = (b-d)(g+h) -> C00 +1
        t = base_t + 6
        U[1][t] = 1; U[3][t] = -1
        V[v_g][t] = 1; V[v_h][t] = 1
        W[w_c00][t] = 1

    # Last column k=8 (Schoolbook)
    # Rank 4
    k = 8
    v_e = k        # B[0, 8]
    v_h = 9 + k    # B[1, 8]
    w_c00 = 2 * k      # C[0, 8]
    w_c01 = 2 * k + 1  # C[1, 8]
    
    base_t = 28
    
    # S1: A00 * B08 -> C08
    t = base_t
    U[0][t] = 1
    V[v_e][t] = 1
    W[w_c00][t] = 1
    
    # S2: A01 * B18 -> C08
    t = base_t + 1
    U[1][t] = 1
    V[v_h][t] = 1
    W[w_c00][t] = 1
    
    # S3: A10 * B08 -> C18
    t = base_t + 2
    U[2][t] = 1
    V[v_e][t] = 1
    W[w_c01][t] = 1
    
    # S4: A11 * B18 -> C18
    t = base_t + 3
    U[3][t] = 1
    V[v_h][t] = 1
    W[w_c01][t] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
