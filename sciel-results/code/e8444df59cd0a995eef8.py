"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M factors (for U)
    # m00, m01, m10, m11
    M_factors = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
    ]
    # N factors (for V)
    # n00, n01, n10, n11
    N_factors = [
        [1, 0, 0, 1],  # N1
        [1, 0, 0, 0],  # N2
        [0, 0, 1, 1],  # N3
        [0, 0, 0, 1],  # N4
        [1, 0, -1, 0], # N5
        [0, 1, 0, -1], # N6
        [-1, 0, 0, 1], # N7
    ]
    
    # W reconstruction coefficients
    # p00, p01, p10, p11
    # P1..P7
    W_factors = [
        [1, 0, 0, 1],  # P1 -> p00, p11
        [0, 1, 1, 0],  # P2 -> p01, p10
        [0, 1, 0, -1], # P3 -> p01, p11 (signs: p01 has -P3, p11 has -P3? No)
        # Let's re-verify Strassen reconstruction
        # p00 = P1 + P4 - P5 + P7
        # p01 = P2 + P3 - P6 + P7
        # p10 = P2 + P4 - P5 + P6
        # p11 = P1 - P3 + P4 + P7
        
        # So for each Pi, we add its coefficient to the corresponding p entries.
        # We will define W_factors as a list of 7 vectors of length 4 (p00, p01, p10, p11)
    ]
    
    # Correct W factors:
    # Index 0: P1. p00+=1, p11+=1
    # Index 1: P2. p01+=1, p10+=1
    # Index 2: P3. p01+=1, p11-=1
    # Index 3: P4. p00+=1, p10+=1, p11+=1
    # Index 4: P5. p00-=1, p10-=1
    # Index 5: P6. p01-=1, p10+=1
    # Index 6: P7. p00+=1, p01+=1, p11+=1
    
    W_coeffs = [
        [1, 0, 0, 1],  # P1
        [0, 1, 1, 0],  # P2
        [0, 1, 0, -1], # P3
        [1, 0, 1, 1],  # P4
        [-1, 0, -1, 0],# P5
        [0, -1, 1, 0], # P6
        [1, 1, 0, 1],  # P7
    ]
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, b_k_start, c_k_start, c_i_start):
        nonlocal col
        # Local indices for A: 0,1 for i; 0,1 for j
        # Local indices for B: 0,1 for j; 0,1 for k
        # Local indices for C: 0,1 for k; 0,1 for i
        
        for c in range(7):
            # Update U
            # A local (li, lj) -> Global (a_i_start + li) * m + (a_j_start + lj)
            for li in range(2):
                for lj in range(2):
                    val = M_factors[c][li * 2 + lj]
                    if val != 0:
                        u_idx = (a_i_start + li) * m + (a_j_start + lj)
                        U[u_idx][col] += val
            
            # Update V
            # B local (lj, lk) -> Global (b_j_start + lj) * p + (b_k_start + lk)
            for lj in range(2):
                for lk in range(2):
                    val = N_factors[c][lj * 2 + lk]
                    if val != 0:
                        v_idx = (b_j_start + lj) * p + (b_k_start + lk)
                        V[v_idx][col] += val
            
            # Update W
            # C local (lk, li) -> Global (c_k_start + lk) * n + (c_i_start + li)
            # W_coeffs[c] is [p00, p01, p10, p11]
            # p00 corresponds to lk=0, li=0
            # p01 corresponds to lk=1, li=0 (since p is 2x2, col 1 is k=1? No, p01 is row 0, col 1. C[i,k]. i=0, k=1.)
            # Wait, W index is k*n + i.
            # p00: i=0, k=0. idx = 0*2 + 0 = 0.
            # p01: i=0, k=1. idx = 1*2 + 0 = 2.
            # p10: i=1, k=0. idx = 0*2 + 1 = 1.
            # p11: i=1, k=1. idx = 1*2 + 1 = 3.
            
            w_vals = W_coeffs[c]
            # Map p00, p01, p10, p11 to local (k, i)
            # p00: k=0, i=0
            # p01: k=1, i=0
            # p10: k=0, i=1
            # p11: k=1, i=1
            
            # w_vals[0] is coeff for p00
            if w_vals[0] != 0:
                w_idx = (c_k_start + 0) * n + (c_i_start + 0)
                W[w_idx][col] += w_vals[0]
            
            if w_vals[1] != 0:
                w_idx = (c_k_start + 1) * n + (c_i_start + 0)
                W[w_idx][col] += w_vals[1]
            
            if w_vals[2] != 0:
                w_idx = (c_k_start + 0) * n + (c_i_start + 1)
                W[w_idx][col] += w_vals[2]
            
            if w_vals[3] != 0:
                w_idx = (c_k_start + 1) * n + (c_i_start + 1)
                W[w_idx][col] += w_vals[3]
            
            col += 1

    # Block 1: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C cols 0,1
    # a_i_start=0, a_j_start=0
    # b_j_start=0, b_k_start=0
    # c_k_start=0, c_i_start=0
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3) -> C cols 2,3
    # a_i_start=0, a_j_start=0
    # b_j_start=0, b_k_start=2
    # c_k_start=2, c_i_start=0
    add_strassen_block(0, 0, 0, 2, 2, 0)
    
    # Block 3: A2 (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C cols 0,1
    # a_i_start=0, a_j_start=2
    # b_j_start=2, b_k_start=0
    # c_k_start=0, c_i_start=0
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3, cols 2,3) -> C cols 2,3
    # a_i_start=0, a_j_start=2
    # b_j_start=2, b_k_start=2
    # c_k_start=2, c_i_start=0
    add_strassen_block(0, 2, 2, 2, 2, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
