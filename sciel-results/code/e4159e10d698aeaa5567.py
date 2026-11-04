"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1]
    ], dtype=object)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, -1, 0, 1, 0, 0, 0],
        [0, 0, 1, -1, 1, 0, 1]
    ], dtype=object)
    
    # W indices: k*2 + i
    # k=0, i=0 -> 0 (C00)
    # k=1, i=0 -> 2 (C10)
    # k=0, i=1 -> 1 (C01)
    # k=1, i=1 -> 3 (C11)
    
    W = np.zeros((4, 7), dtype=object)
    # M1 -> C00
    W[0, 0] = 1
    # M2 -> C10
    W[2, 1] = 1
    # M3 -> C01
    W[1, 2] = 1
    # M4 -> C10
    W[2, 3] = 1
    # M5 -> C11
    W[3, 4] = 1
    # M6 -> C00, C10
    W[0, 5] = 1
    W[2, 5] = 1
    # M7 -> C01, C11
    W[1, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _validate_decomposition(U, V, W, n, m, p):
    """
    Checks if U @ diag(s) @ V^T equals the multiplication tensor.
    We just need to check that for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if s != 1:
                    return False
    return True

def _fraction_to_float_or_int(f):
    if f == 0:
        return 0
    if f.denominator == 1:
        return int(f.numerator)
    return float(f)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0 - 5.0 # 145s safety margin
    
    U_base, V_base, W_base = _strassen_2x2x2()
    
    # Split into blocks
    # n=2, m=4, p=6
    # i blocks: 2 (size 1)
    # j blocks: 2 (size 2)
    # k blocks: 3 (size 2)
    
    i_blocks = [(0, 1), (1, 2)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    # Total rank = 2 * 2 * 3 * 7 = 84
    total_rank = len(i_blocks) * len(j_blocks) * len(k_blocks) * 7
    
    U = np.zeros((n * m, total_rank), dtype=object)
    V = np.zeros((m * p, total_rank), dtype=object)
    W = np.zeros((p * n, total_rank), dtype=object)
    
    rank_offset = 0
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                # Extract sub-blocks of base matrices
                # U_base rows correspond to i_local (0,1), j_local (0,1)
                # Here i_local is always 0 because block size is 1?
                # Wait, i block size is 1. So i_local is 0.
                # j block size is 2. j_local is 0, 1.
                # k block size is 2. k_local is 0, 1.
                
                # The base Strassen matrices are defined for 2x2x2.
                # Our block is 1x2x2.
                # This is not a 2x2x2 block.
                # We need to handle this carefully.
                # Strassen is for 2x2x2.
                # If we have 1x2x2, we can just use schoolbook for that slice?
                # Or we can pad the 1x2x2 to 2x2x2?
                # If we pad, we add zeros to A, B, C.
                # Multiplication of padded matrices is same as original for the non-zero part.
                # So we can treat each (1, 2, 2) block as a (2, 2, 2) block with the second row/col zeroed out?
                # No, A is 1x2. B is 2x2. C is 2x1.
                # If we pad A to 2x2, B to 2x2, C to 2x2:
                # A_pad = [[A], [0]]
                # B_pad = B
                # C_pad = [[C], [0]]
                # A_pad @ B_pad = [[C], [0]] = C_pad.
                # So yes, we can use the 2x2x2 Strassen decomposition for the padded block.
                # The terms involving the zeroed out parts will be zero.
                # So we can just use the full 7 terms.
                
                # Map local indices to global
                # i_local in 0..1 -> i_global = i_start + i_local
                # But i_block size is 1. So only i_local=0 is used?
                # Wait, if I use the 2x2x2 Strassen, it assumes 2x2 inputs.
                # My input is 1x2.
                # So I am effectively doing 2x2x2 multiplication where A has 0 in row 1.
                # The result C will have 0 in row 1.
                # But I only care about row 0 of C.
                # So I can just take the 7 terms and map them.
                
                for c_local in range(7):
                    c_global = rank_offset + c_local
                    
                    # U contributions
                    for i_local in range(2):
                        for j_local in range(2):
                            val = U_base[i_local * 2 + j_local, c_local]
                            if val != 0:
                                i_global = i_start + i_local
                                j_global = j_start + j_local
                                if i_global < n and j_global < m:
                                    U[i_global * m + j_global, c_global] += val
                                    
                    # V contributions
                    for j_local in range(2):
                        for k_local in range(2):
                            val = V_base[j_local * 2 + k_local, c_local]
                            if val != 0:
                                j_global = j_start + j_local
                                k_global = k_start + k_local
                                if j_global < m and k_global < p:
                                    V[j_global * p + k_global, c_global] += val
                                    
                    # W contributions
                    for k_local in range(2):
                        for i_local in range(2):
                            val = W_base[k_local * 2 + i_local, c_local]
                            if val != 0:
                                k_global = k_start + k_local
                                i_global = i_start + i_local
                                if k_global < p and i_global < n:
                                    W[k_global * n + i_global, c_global] += val
                                  
                rank_offset += 7
                
    # Convert to lists of ints/floats
    U_out = []
    for row in U:
        U_out.append([_fraction_to_float_or_int(x) for x in row])
    V_out = []
    for row in V:
        V_out.append([_fraction_to_float_or_int(x) for x in row])
    W_out = []
    for row in W:
        W_out.append([_fraction_to_float_or_int(x) for x in row])
        
    # Verify
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_school = [[0] * rank for _ in range(n * m)]
        V_school = [[0] * rank for _ in range(m * p)]
        W_school = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
