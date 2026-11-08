"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We use a 4/2 split for 6x6 matrices.
    # A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x2, A21 is 2x4, A22 is 2x2.
    # B = [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x2, B21 is 2x4, B22 is 2x2.
    # C = A B.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We decompose each term using Strassen for 2x2 blocks.
    # Total rank is sum of ranks of these terms.
    
    # Helper to add a 2x2 Strassen decomposition
    # off_i, off_j, off_k are the top-left offsets in A, B, C
    # rank_offset is the starting column index in U, V, W
    # Returns new rank_offset
    
    def add_strassen_2x2(off_i, off_j, off_k, rank_offset, U, V, W):
        # Strassen patterns for 2x2 matrices
        # (u_coeffs, v_coeffs, w_coeffs)
        # coeffs are dicts mapping (row, col) to value
        strassen_patterns = [
            # M0 = (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M1 = (A10+A11)B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M2 = A00(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M3 = A11(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M4 = (A00+A01)B11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M5 = (A10-A00)(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M6 = (A01-A11)(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            if time.time() > deadline:
                break
            
            # Update U: A is n x m, flattened as row-major: index = i*m + j
            for (pi, pj), val in u_coeffs.items():
                r = off_i + pi
                c = off_j + pj
                U[r * m + c][rank_offset] += val
            
            # Update V: B is m x p, flattened as row-major: index = j*p + k
            for (pi, pj), val in v_coeffs.items():
                r = off_j + pi
                c = off_k + pj
                V[r * p + c][rank_offset] += val
            
            # Update W: C is p x n (output), index is k*n + i per trace convention
            for (pi, pj), val in w_coeffs.items():
                i = off_i + pi
                k = off_k + pj
                W[k * n + i][rank_offset] += val
            
            rank_offset += 1
            
        return rank_offset

    # Total rank estimation
    # M444: 7 * 7 = 49
    # M422: 2 * 7 = 14
    # M442: 4 * 7 = 28
    # M422: 2 * 7 = 14
    # M244: 4 * 7 = 28
    # M224: 4 * 7 = 28
    # M242: 2 * 7 = 14
    # M222: 1 * 7 = 7
    # Total: 182
    
    total_rank = 182
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # 1. A11 B11 (4x4 * 4x4)
    # Decompose into 7 2x2 multiplications using Strassen on 2x2 blocks of 4x4
    # Offsets:
    # A11 is at (0,0) in A.
    # B11 is at (0,0) in B.
    # C11 is at (0,0) in C.
    
    # Strassen for 4x4:
    # M0 = (A00+A11)(B00+B11) -> 2x2 at (0,0) and (2,2)
    # This is complex to map directly.
    # Instead, we can just call add_strassen_2x2 for each of the 7 components?
    # No, Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # The 2x2 matrices are linear combinations of the 2x2 blocks of the 4x4 matrices.
    # This is what add_strassen_2x2 does, but it assumes the input is 2x2.
    # For 4x4, we need to apply Strassen to the 2x2 blocks.
    
    # Let's implement a recursive call or just hardcode the 4x4 Strassen.
    # Actually, the current add_strassen_2x2 adds ONE 2x2 decomposition.
    # For M444, we need 7 such decompositions? No, M444 is ONE Strassen step on 2x2 blocks.
    # The 7 multiplications are of 2x2 matrices.
    # So we need to define the 2x2 matrices that are multiplied.
    
    # This is getting too complex to hardcode manually in the time available.
    # I will stick to the 189 solution but ensure it's correct.
    # Wait, I can just use the 189 solution.
    
    # Reverting to the 189 solution as it is safer.
    
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
                    if time.time() > deadline:
                        break
                    
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    for (pi, pj), val in w_coeffs.items():
                        i = I * block_size + pi
                        k = K * block_size + pj
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
