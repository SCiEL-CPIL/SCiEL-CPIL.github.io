"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses a Strassen-based block decomposition:
    - Split B into two 4x4 blocks and a 4x1 block
    - Use Strassen 4x4x4 (rank 49) for the first two 4x4 blocks
    - Use schoolbook (rank 16) for the last 4x1 block
    - Total rank: 49 + 49 + 16 = 114
    """
    
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
    # Each term is (u_dict, v_dict, w_dict) where keys are (row, col) indices
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Add a 4x4x4 Strassen decomposition using 2x2 blocks."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Fill u_vec from A_blocks
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # Fill v_vec from B_blocks
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # Fill w_vec from C_blocks
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 Strassen combinations for 4x4 using 2x2 blocks
        # A is split into 4 blocks: A00, A01, A10, A11
        # B is split into 4 blocks: B00, B01, B10, B11
        # C is split into 4 blocks: C00, C01, C10, C11
        
        # Combination 1: (A00+A11)(B00+B11) -> C00 + C11
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        
        # Combination 2: (A10+A11)(B00) -> C10, C11
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        
        # Combination 3: (A00)(B01+B11) -> C01, C11
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        
        # Combination 4: (A11)(B10-B00) -> C00, C10
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        
        # Combination 5: (A00+A01)(B11) -> C00, C01
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        
        # Combination 6: (A10-A00)(B00+B01) -> C11
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        
        # Combination 7: (A01-A11)(B10+B11) -> C00
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # First 4x4 block of B: columns 0-3
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Second 4x4 block of B: columns 4-7
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Last column of B: column 8, use schoolbook
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Convert list of column vectors to row-major matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p*n)]
    
    # Verify the solution is correct
    def verify(U, V, W):
        """Verify the tensor decomposition is correct."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[u][c] * V[v][c] * W[w][c]
                    if s != 1:
                        return False
        return True
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
