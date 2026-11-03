"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140.0

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Use Strassen's algorithm for the 4x4 * 4x4 part (cols 0-3 of B): rank 49
    - Use schoolbook for the 4x4 * 4x5 part (cols 4-8 of B): rank 80
    - Total rank = 49 + 80 = 129
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 2x2*2x2 rank-7 Strassen decomposition terms
    # For X=[[x0,x1],[x2,x3]], Y=[[y0,y1],[y2,y3]], Z=X*Y
    # z0 = M1+M4-M5+M7, z1 = M3+M5, z2 = M2+M4, z3 = M1-M2+M3+M6
    # Row-major: X = [x0, x1, x2, x3], Y = [y0, y1, y2, y3], Z = [z0, z1, z2, z3]
    s22_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),    # M1: (x0+x3)(y0+y3)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),    # M2: (x2+x3)(y0)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # M3: x0*(y1-y3)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),   # M4: x3*(y2-y0)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5: (x0+x1)(y3)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6: (x2-x0)(y0+y1)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),   # M7: (x1-x3)(y2+y3)
    ]
    
    def build_4x4_strassen():
        """
        Returns (U, V, W) for 4x4 * 4x4 using Strassen.
        U: 16 x 49, V: 16 x 49, W: 16 x 49
        W convention: W[k*4+i, r] for C[i,k]
        """
        # 7 Strassen block products for 4x4*4x4
        block_products = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),   # S1 = (A11+A22)*(B11+B22)
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),               # S2 = (A21+A22)*B11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),              # S3 = A11*(B12-B22)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),              # S4 = A22*(B21-B11)
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),               # S5 = (A11+A12)*B22
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),    # S6 = (A21-A11)*(B11+B12)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),    # S7 = (A12-A22)*(B21+B22)
        ]
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        def build_vec(block_coeffs):
            """block_coeffs: list of 4 coefficients for blocks (0,0),(0,1),(1,0),(1,1)"""
            vec = [0]*16
            for pos, coeff in enumerate(block_coeffs):
                if coeff != 0:
                    br, bc = pos // 2, pos % 2
                    for dr in range(2):
                        for dc in range(2):
                            idx = (2*br + dr) * 4 + (2*bc + dc)
                            vec[idx] += coeff
            return vec
        
        term_idx = 0
        for si, (x_blocks, y_blocks) in enumerate(block_products):
            x_coeff = [0]*4
            for (br,bc), c in x_blocks.items():
                x_coeff[br*2+bc] = c
            y_coeff = [0]*4
            for (br,bc), c in y_blocks.items():
                y_coeff[br*2+bc] = c
            
            for xv, yv, zv in s22_terms:
                x_combined = [xv[pos] * x_coeff[pos] for pos in range(4)]
                y_combined = [yv[pos] * y_coeff[pos] for pos in range(4)]
                z_combined = list(zv)
                
                U_vec = build_vec(x_combined)
                V_vec = build_vec(y_combined)
                
                W_vec = [0]*16
                for pos, coeff in enumerate(z_combined):
                    if coeff != 0:
                        br, bc = pos // 2, pos % 2
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*br + dr
                                k = 2*bc + dc
                                W_idx = k * 4 + i
                                W_vec[W_idx] += coeff
                
                for i in range(16):
                    U[i][term_idx] = U_vec[i]
                    V[i][term_idx] = V_vec[i]
                    W[i][term_idx] = W_vec[i]
                term_idx += 1
        
        return U, V, W
    
    U44, V44, W44 = build_4x4_strassen()
    
    # Schoolbook for 4x4 * 4x5 (B2 is columns 4-8 of B)
    rank_sb = 80
    U_sb = [[0]*rank_sb for _ in range(16)]
    V_sb = [[0]*rank_sb for _ in range(20)]  # 4*5
    W_sb = [[0]*rank_sb for _ in range(36)]  # 9*4
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                r = i * 20 + j * 5 + k
                U_sb[i*4 + j][r] = 1
                V_sb[j*5 + k][r] = 1  # k is 0..4, corresponds to cols 4..8 of B
                W_sb[(k+4)*4 + i][r] = 1  # col index in full B is k+4
    
    # Combine: total rank = 49 + 80 = 129
    total_rank = 49 + 80
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]  # 4*9
    W = [[0]*total_rank for _ in range(36)]  # 9*4
    
    # Copy Strassen part (for B cols 0-3)
    for r in range(49):
        for i in range(16):
            U[i][r] = U44[i][r]
        for j in range(16):
            V[j][r] = V44[j][r]
        for k in range(16):
            W[k][r] = W44[k][r]
    
    # Copy schoolbook part (for B cols 4-8)
    for r in range(80):
        for i in range(16):
            U[i][49 + r] = U_sb[i][r]
        for j in range(20):
            V[j][49 + r] = V_sb[j][r]
        for k in range(36):
            W[k][49 + r] = W_sb[k][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
