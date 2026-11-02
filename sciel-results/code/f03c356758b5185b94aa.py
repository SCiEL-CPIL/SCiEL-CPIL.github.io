"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x12 multiplication into 6 independent 2x2 multiplications.
    # Each 2x2 multiplication uses Strassen's algorithm with rank 7.
    # Total rank = 6 * 7 = 42.
    rank = 42
    num_blocks = p // 2  # 6 blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # For each block u (0..5) and each Strassen term t (0..6)
    for u in range(num_blocks):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V
            # B indices for block u: 
            # B is 2x12. Block u corresponds to columns 2u and 2u+1.
            # B row 0, col 2u: index 0*p + 2u
            # B row 0, col 2u+1: index 0*p + 2u + 1
            # B row 1, col 2u: index 1*p + 2u
            # B row 1, col 2u+1: index 1*p + 2u + 1
            
            b00_idx = 0 * p + 2 * u
            b01_idx = 0 * p + 2 * u + 1
            b10_idx = 1 * p + 2 * u
            b11_idx = 1 * p + 2 * u + 1
            
            if t == 0:
                V[b00_idx][c] += 1
                V[b11_idx][c] += 1
            elif t == 1:
                V[b00_idx][c] += 1
            elif t == 2:
                V[b01_idx][c] += 1
                V[b11_idx][c] -= 1
            elif t == 3:
                V[b10_idx][c] += 1
                V[b00_idx][c] -= 1
            elif t == 4:
                V[b11_idx][c] += 1
            elif t == 5:
                V[b00_idx][c] += 1
                V[b01_idx][c] += 1
            elif t == 6:
                V[b10_idx][c] += 1
                V[b11_idx][c] += 1
                
            # Fill W
            # Output C is 2x12.
            # W index convention: W[k*n + i, r]
            # k is column of C (0..11), i is row of C (0..1).
            # Block u corresponds to columns 2u and 2u+1.
            # C[0, 2u] index: (2u)*n + 0
            # C[0, 2u+1] index: (2u+1)*n + 0
            # C[1, 2u] index: (2u)*n + 1
            # C[1, 2u+1] index: (2u+1)*n + 1
            
            c00_idx = (2 * u) * n + 0
            c01_idx = (2 * u + 1) * n + 0
            c10_idx = (2 * u) * n + 1
            c11_idx = (2 * u + 1) * n + 1
            
            if t == 0:
                W[c00_idx][c] += 1
                W[c11_idx][c] += 1
            elif t == 1:
                W[c10_idx][c] += 1
                W[c11_idx][c] -= 1
            elif t == 2:
                W[c01_idx][c] += 1
                W[c11_idx][c] += 1
            elif t == 3:
                W[c00_idx][c] += 1
                W[c10_idx][c] += 1
            elif t == 4:
                W[c00_idx][c] -= 1
                W[c01_idx][c] += 1
            elif t == 5:
                W[c11_idx][c] += 1
            elif t == 6:
                W[c00_idx][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
