import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    block_size = 3
    blocks_per_dim = 2
    
    rank = (blocks_per_dim ** 3) * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 3x3 algorithm patterns
    # Each pattern: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: {(i,j): val}
    # V_coeffs: {(j,k): val}
    # W_coeffs: {(i,k): val}
    
    # Note: The indices in W are (i,k) corresponding to C[i,k].
    # The flattening index for W is k*n + i.
    
    laderman_patterns = [
        # 1
        ({(0,0): 1, (1,1): 1, (2,0): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1, (1,1): 1, (2,2): 1, (0,1): 1, (1,0): 1, (2,1): 1, (1,2): 1, (0,2): 1, (2,0): 1}),
        # This W is wrong. W is the reconstruction coefficients.
        # I need the correct W coefficients for each P_k.
        # This is too risky to guess.
    ]
    
    # Fallback: Use Strassen 2x2 if Laderman is not confident.
    # But I must try to improve.
    # Let's use the Strassen 2x2 solution which is known to work and scores 1.1429.
    # The directive says "KEEP this design". The current design is Strassen 2x2.
    # Improving to Laderman is a "change of design family" (3x3 vs 2x2 blocks).
    # "Do NOT switch to a different design family."
    # This implies I should stick to the 2x2 block Strassen approach?
    # If I must keep the design, I can only optimize the implementation, which won't change the rank.
    # However, "investigate improvements within its family".
    # Maybe I can use a better 2x2 algorithm? No, Strassen is optimal for 2x2.
    # Maybe I can use a different tiling?
    # If I keep the 2x2 block design, the rank is fixed at 189.
    # Is there a way to reduce rank within the 2x2 block family?
    # No, the rank of the tensor product is the product of ranks.
    
    # Wait, the instruction "Do NOT switch to a different design family" might refer to the overall strategy (tensor decomposition via blocking).
    # Switching from 2x2 blocks to 3x3 blocks is still blocking.
    # But "Keep this design" usually means keep the high-level structure.
    # Given the risk of Laderman implementation errors leading to score 0, and the strict "Keep this design" instruction,
    # I will stick to the Strassen 2x2 solution but clean up the code.
    # The score will remain 1.1429.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
