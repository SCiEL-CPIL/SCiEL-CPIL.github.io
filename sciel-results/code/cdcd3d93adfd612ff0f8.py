"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-28 exact decomposition of the (2,4,4) matrix multiplication tensor
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS # 2, 4, 4
    
    # We partition A (2x4) into A1 (2x2) and A2 (2x2)
    # We partition B (4x4) into B11, B12 (top 2 rows) and B21, B22 (bottom 2 rows)
    # Specifically:
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C[:, 0:2] = A1 B11 + A2 B21
    # C[:, 2:4] = A1 B12 + A2 B22
    
    # We use Strassen's 7-term algorithm for each 2x2 matrix multiplication.
    # There are 4 such multiplications: P1=A1B11, P2=A2B21, P3=A1B12, P4=A2B22.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank)) # 8 x 28
    V = np.zeros((m * p, rank)) # 16 x 28
    W = np.zeros((p * n, rank)) # 8 x 28
    
    # Strassen terms for X (2x2) * Y (2x2):
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    
    # Result Z = X*Y:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def add_strassen_term(offset, X_idx_map, Y_idx_map, Z_idx_map):
        """
        Adds 7 Strassen terms for a 2x2 multiplication.
        X_idx_map: dict mapping local X indices (0,1,2,3) to global U indices
                   X is 2x2, flattened row-major: 00, 01, 10, 11 -> 0, 1, 2, 3
                   Global U index for A[i, j] is i*m + j.
        Y_idx_map: dict mapping local Y indices to global V indices
                   Y is 2x2, flattened row-major.
                   Global V index for B[j, k] is j*p + k.
        Z_idx_map: dict mapping local Z indices to global W indices
                   Z is 2x2, flattened row-major.
                   Global W index for C[k, i] is k*n + i. 
                   Note: W[k*n+i, r]. So Z00 corresponds to C[0,0] -> W[0*2+0]=W[0].
                         Z01 corresponds to C[1,0] -> W[1*2+0]=W[2]? 
                         Wait. C is 2x4. C[i, k].
                         W index is k*n + i.
                         If Z is 2x2 block of C, say C[:, 0:2], then:
                         Z00 = C[0,0] -> W[0*2+0] = W[0]
                         Z01 = C[1,0] -> W[0*2+1] = W[1]
                         Z10 = C[0,1] -> W[1*2+0] = W[2]
                         Z11 = C[1,1] -> W[1*2+1] = W[3]
                         Wait, C[i, k]. i is row, k is col.
                         Z is 2x2. Z[r, c] = C[i_base + r, k_base + c].
                         W index = (k_base + c) * n + (i_base + r).
        """
        # Strassen coefficients for U (X), V (Y), W (Z)
        # We define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs is a list of 4 values for X00, X01, X10, X11
        # v_coeffs is a list of 4 values for Y00, Y01, Y10, Y11
        # w_coeffs is a list of 4 values for Z00, Z01, Z10, Z11 (contribution to Z)
        
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # Z00 += 1, Z11 += 1
        
        # M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 0] # Z10 += 1
        
        # M3
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, 0] # Z01 += 1
        
        # M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1] # Y10 - Y00
        w4 = [1, 0, 1, 0] # Z00 += 1, Z10 += 1
        
        # M5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 0, 0, 1] # Z00 -= 1, Z11 += 1
        
        # M6
        u6 = [0, 0, 1, -1] # X10 - X00
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # Z11 += 1
        
        # M7
        u7 = [0, 1, 0, -1] # X01 - X11
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0] # Z00 += 1
        
        terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for t in range(7):
            col = offset + t
            u_c, v_c, w_c = terms[t]
            
            # Update U
            for local_idx, val in enumerate(u_c):
                if val != 0:
                    global_idx = X_idx_map[local_idx]
                    U[global_idx, col] += val
                    
            # Update V
            for local_idx, val in enumerate(v_c):
                if val != 0:
                    global_idx = Y_idx_map[local_idx]
                    V[global_idx, col] += val
                    
            # Update W
            for local_idx, val in enumerate(w_c):
                if val != 0:
                    global_idx = Z_idx_map[local_idx]
                    W[global_idx, col] += val

    # Define index mappings for the 4 sub-problems
    
    # Problem 1: A1 * B11
    # A1 is A[:, 0:2]. A1[0,0]=A[0,0], A1[0,1]=A[0,1], A1[1,0]=A[1,0], A1[1,1]=A[1,1]
    # U indices:
    # A[0,0] -> 0*4+0 = 0
    # A[0,1] -> 0*4+1 = 1
    # A[1,0] -> 1*4+0 = 4
    # A[1,1] -> 1*4+1 = 5
    X1_map = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # B11 is B[0:2, 0:2].
    # V indices:
    # B[0,0] -> 0*4+0 = 0
    # B[0,1] -> 0*4+1 = 1
    # B[1,0] -> 1*4+0 = 4
    # B[1,1] -> 1*4+1 = 5
    Y1_map = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # Result Z1 contributes to C[:, 0:2] (first 2 cols of C)
    # C[0,0] -> W[0*2+0]=0
    # C[0,1] -> W[1*2+0]=2
    # C[1,0] -> W[0*2+1]=1
    # C[1,1] -> W[1*2+1]=3
    # Z local: 00->C[0,0], 01->C[0,1], 10->C[1,0], 11->C[1,1]
    # Wait, Z[r,c] = C[i_base+r, k_base+c].
    # Here i_base=0, k_base=0.
    # Z00 = C[0,0] -> W[0]
    # Z01 = C[0,1] -> W[2]
    # Z10 = C[1,0] -> W[1]
    # Z11 = C[1,1] -> W[3]
    Z1_map = {0: 0, 1: 2, 2: 1, 3: 3}
    
    add_strassen_term(0, X1_map, Y1_map, Z1_map)
    
    # Problem 2: A2 * B21
    # A2 is A[:, 2:4].
    # A2[0,0]=A[0,2] -> 0*4+2=2
    # A2[0,1]=A[0,3] -> 0*4+3=3
    # A2[1,0]=A[1,2] -> 1*4+2=6
    # A2[1,1]=A[1,3] -> 1*4+3=7
    X2_map = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # B21 is B[2:4, 0:2].
    # B[2,0] -> 2*4+0=8
    # B[2,1] -> 2*4+1=9
    # B[3,0] -> 3*4+0=12
    # B[3,1] -> 3*4+1=13
    Y2_map = {0: 8, 1: 9, 2: 12, 3: 13}
    
    # Result Z2 contributes to C[:, 0:2] (same as Z1)
    # So Z2_map is same as Z1_map
    Z2_map = Z1_map
    
    add_strassen_term(7, X2_map, Y2_map, Z2_map)
    
    # Problem 3: A1 * B12
    # A1 is same as before.
    X3_map = X1_map
    
    # B12 is B[0:2, 2:4].
    # B[0,2] -> 0*4+2=2
    # B[0,3] -> 0*4+3=3
    # B[1,2] -> 1*4+2=6
    # B[1,3] -> 1*4+3=7
    Y3_map = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # Result Z3 contributes to C[:, 2:4] (last 2 cols of C)
    # i_base=0, k_base=2.
    # Z00 = C[0,2] -> W[2*2+0]=4
    # Z01 = C[0,3] -> W[3*2+0]=6
    # Z10 = C[1,2] -> W[2*2+1]=5
    # Z11 = C[1,3] -> W[3*2+1]=7
    Z3_map = {0: 4, 1: 6, 2: 5, 3: 7}
    
    add_strassen_term(14, X3_map, Y3_map, Z3_map)
    
    # Problem 4: A2 * B22
    # A2 is same as before.
    X4_map = X2_map
    
    # B22 is B[2:4, 2:4].
    # B[2,2] -> 2*4+2=10
    # B[2,3] -> 2*4+3=11
    # B[3,2] -> 3*4+2=14
    # B[3,3] -> 3*4+3=15
    Y4_map = {0: 10, 1: 11, 2: 14, 3: 15}
    
    # Result Z4 contributes to C[:, 2:4] (same as Z3)
    Z4_map = Z3_map
    
    add_strassen_term(21, X4_map, Y4_map, Z4_map)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
