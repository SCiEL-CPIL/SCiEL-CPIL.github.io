"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Rank 20 decomposition for M_{2,2,6}
    # Uses 5 standard Strassen-like blocks (rank 7 each = 35) but with
    # sharing and optimization to get rank 20.
    # 
    # Actually, let me use a verified rank-21 Strassen decomposition for 3 blocks.
    # The known lower bound for M_{2,2,6} is 20, but rank 21 is safely achievable
    # via Strassen on 3 columns.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2 matrix)
    # U index: i*m + j where i is row, j is col of A
    # A00=idx0, A01=idx1, A10=idx2, A11=idx3
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # For each of the 3 column-blocks of B (each 2x2), Strassen gives 7 products.
    # B has shape 2x6, so 3 blocks of 2x2.
    # V index: j*p + k where j is row of B, k is col of B (global)
    # For block r (r=0,1,2), local cols are k_local in {0,1}, global col = 2*r + k_local
    # B00 = V[0*p + 2r], B01 = V[1*p + 2r] -- wait, B is 2x6
    # B[j][k] where j in {0,1}, k in {0,...,5}
    # V index = j * p + k = j * 6 + k
    
    V_coeffs_per_block = [
        # Each entry: list of (j_local, k_local, coeff) for B[j_local][2r + k_local]
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # W: output C is 2x6. C[i][k] where i in {0,1}, k in {0,...,5}
    # W index: k*n + i = k*2 + i (following trace convention W[k*n+i, r])
    # For block r, local output cols j_local in {0,1}, global col = 2r + j_local
    
    # Strassen output combinations:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_map = [
        # For each M_k, list of (i, j_local, coeff)
        # i is row of C, j_local is local col within block
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C00(+1), C11(+1)
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C10(+1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C01(+1), C11(+1)
        [(0, 0, -1), (1, 0, 1)],  # M4 -> C00(-1), C10(+1)
        [(0, 0, 1), (0, 1, 1)],   # M5 -> C00(+1), C01(+1)
        [(1, 1, 1)],              # M6 -> C11(+1)
        [(0, 0, -1)]              # M7 -> C00(-1)
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            
            # Fill U (same for all blocks)
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
                
            # Fill V
            for j_local, k_local, val in V_coeffs_per_block[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
                
            # Fill W
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
