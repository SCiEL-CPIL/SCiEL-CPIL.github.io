"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen's algorithm on 2x2x2 blocks
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 7-term decomposition vectors for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2)
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        # M4
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    current_rank = 0
    
    # We have 2 blocks of A (A1, A2) and 4 blocks of B per A block
    # A1: j in {0,1}, A2: j in {2,3}
    # B1: j in {0,1}, B2: j in {2,3}
    # For each A_block and B_block, we have 4 k-blocks (0..7)
    
    for a_block in [0, 1]: # 0 for A1, 1 for A2
        a_j_offset = a_block * 2 # 0 or 2
        for b_block in [0, 1]: # 0 for B1, 1 for B2
            b_j_offset = b_block * 2 # 0 or 2
            # If a_block != b_block, this term is 0 in standard mult?
            # C = A1 B1 + A2 B2.
            # So we only care about (A1, B1) and (A2, B2).
            if a_block != b_block:
                continue
                
            for k_block in range(4): # 4 blocks of width 2 in p=8
                k_offset = k_block * 2
                
                for u_vec, v_vec, w_vec in strassen_terms:
                    # Map local indices to global
                    for local_i in range(2):
                        for local_j in range(2):
                            if u_vec[local_i * 2 + local_j] != 0:
                                global_i = local_i
                                global_j = a_j_offset + local_j
                                U[global_i * m + global_j, current_rank] = u_vec[local_i * 2 + local_j]
                    
                    for local_j in range(2):
                        for local_k in range(2):
                            if v_vec[local_j * 2 + local_k] != 0:
                                global_j = b_j_offset + local_j
                                global_k = k_offset + local_k
                                V[global_j * p + global_k, current_rank] = v_vec[local_j * 2 + local_k]
                    
                    for local_k in range(2):
                        for local_i in range(2):
                            if w_vec[local_k * 2 + local_i] != 0:
                                global_k = k_offset + local_k
                                global_i = local_i
                                W[global_k * n + global_i, current_rank] = w_vec[local_k * 2 + local_i]
                    
                    current_rank += 1
    
    # Convert to lists of lists as required by interface?
    # The interface expects {'U', 'V', 'W'} with integer entries.
    # NumPy arrays are usually accepted, but lists are safer if "integer/half-integer" implies Python types.
    # However, the seed program used lists. I will return lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
