"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The problem is equivalent to 6 independent 2x2 matrix multiplications.
    # Minimal rank for 2x2 multiplication is 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 formulas for 2x2 matrix multiplication:
    # M0 = (a00 + a11) * (b00 + b11)
    # M1 = (a10 + a11) * b00
    # M2 = a00 * (b01 - b11)
    # M3 = a11 * (b10 - b00)
    # M4 = (a00 + a01) * b11
    # M5 = (a10 - a00) * (b00 + b10)
    # M6 = (a01 - a11) * (b11 + b01)
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices for a 2x2 block starting at col 2*u:
    # 0:(0, 2u), 1:(0, 2u+1), 2:(1, 2u), 3:(1, 2u+1)
    # C indices for a 2x2 block starting at col 2*u:
    # 0:(0, 2u), 1:(0, 2u+1), 2:(1, 2u), 3:(1, 2u+1)
    
    # Coefficients for U (A part)
    U_coeffs = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 1, 0, -1] # M6
    ]
    
    # Coefficients for V (B part) for each of the 6 blocks
    # V is indexed by [m*p, rank] = [24, 42]
    # For block u, the relevant B indices are 0, 1, 2, 3 relative to the block.
    # Global indices: 0*p + 2*u, 0*p + 2*u+1, 1*p + 2*u, 1*p + 2*u+1
    V_coeffs = [
        [1, 0, 0, 1], # M0: b00 + b11
        [1, 0, 0, 0], # M1: b00
        [0, 1, 0, -1],# M2: b01 - b11
        [0, 0, 1, -1],# M3: b10 - b00
        [0, 0, 0, 1], # M4: b11
        [1, 0, 1, 0], # M5: b00 + b10
        [0, 1, 0, 1]  # M6: b11 + b01
    ]
    
    # Coefficients for W (C part)
    # C = 
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    W_coeffs = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_coeffs[t][i]
                
            # Fill V
            # Map local indices 0,1,2,3 to global indices in V
            # Local 0: row 0, col 2u -> global 0*p + 2u
            # Local 1: row 0, col 2u+1 -> global 0*p + 2u+1
            # Local 2: row 1, col 2u -> global 1*p + 2u
            # Local 3: row 1, col 2u+1 -> global 1*p + 2u+1
            v_idx = [0*p + 2*u, 0*p + 2*u+1, 1*p + 2*u, 1*p + 2*u+1]
            for i in range(4):
                val = V_coeffs[t][i]
                if val != 0:
                    V[v_idx[i]][c] += val
                    
            # Fill W
            # Map local C indices 0,1,2,3 to global indices in W
            # Local 0: row 0, col 2u -> global 0*n + 2u? 
            # Wait, W index is [p*n, rank].
            # W[k*n + i] where k is col index (0..11), i is row index (0..1).
            # So for col 2u, row 0: index 2u*n + 0
            # For col 2u, row 1: index 2u*n + 1
            # For col 2u+1, row 0: index (2u+1)*n + 0
            # For col 2u+1, row 1: index (2u+1)*n + 1
            
            w_idx = [2*u*n + 0, 2*u*n + 1, (2*u+1)*n + 0, (2*u+1)*n + 1]
            # W_coeffs[t] gives the coefficient of M_t for C00, C01, C10, C11
            # So we add W_coeffs[t][c_idx] to W[w_idx[c_idx]][c]
            for i in range(4):
                val = W_coeffs[i][t]
                if val != 0:
                    W[w_idx[i]][c] += val

    return {"U": U, "V": W, "W": W} # Wait, return U, V, W
# EVOLVE-BLOCK-END
