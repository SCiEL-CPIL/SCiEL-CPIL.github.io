"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses a block-wise approach:
    - Partition dimensions 3, 5, 5 into blocks of size 2 and 1.
    - Apply Strassen's algorithm (rank 7) to 2x2x2 blocks.
    - Apply schoolbook algorithm (rank abc) to other blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    
    # Partitions
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=5 -> [2, 2, 1]
    
    I_parts = [(0, 2), (2, 3)]  # (start, end)
    J_parts = [(0, 2), (2, 4), (4, 5)]
    K_parts = [(0, 2), (2, 4), (4, 5)]
    
    # Initialize matrices
    # We don't know the total rank yet, so we'll collect terms in a list
    # Term: (i, j, k, u_val, v_val, w_val)
    # But we need to map to global indices.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Let's just build the matrices with a large enough size or dynamically.
    # Max rank is 75. We expect ~71.
    # Let's use a list of tuples (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # Then we can aggregate if needed, but since each term is distinct in rank,
    # we just need to assign column indices.
    
    terms = []
    
    def add_term(i, j, k, u_val, v_val, w_val):
        if u_val == 0 and v_val == 0 and w_val == 0:
            return
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))

    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (A) coefficients for each M
    # U[i,j, r]
    Strassen_U = [
        [1, 0, 0, 0, 1, 0, 0], # M1: A00=1, A11=1 -> U[0,0]=1, U[1,1]=1
        [0, 0, 0, 0, 0, 0, 0], # Placeholder, will fill properly
    ]
    # Let's define them explicitly for clarity
    # r=0 (M1): A00=1, A11=1
    # r=1 (M2): A10=1, A11=1
    # r=2 (M3): A00=1
    # r=3 (M4): A11=1
    # r=4 (M5): A00=1, A01=1
    # r=5 (M6): A10=1, A00=-1
    # r=6 (M7): A01=1, A11=-1
    
    S_U = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    # V (B) coefficients
    # r=0 (M1): B00=1, B11=1
    # r=1 (M2): B00=1
    # r=2 (M3): B01=1, B11=-1
    # r=3 (M4): B10=1, B00=-1
    # r=4 (M5): B11=1
    # r=5 (M6): B00=1, B01=1
    # r=6 (M7): B10=1, B11=1
    
    S_V = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    # W (C) coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    S_W = [
        # r=0 (M1): C00=1, C11=1
        { (0,0): 1, (1,1): 1 }, # W[k*n+i] -> W[0,0] and W[1,1]
        # r=1 (M2): C10=1, C11=-1
        { (0,1): 1, (1,1): -1 },
        # r=2 (M3): C01=1, C11=1
        { (1,0): 1, (1,1): 1 },
        # r=3 (M4): C00=1, C10=1
        { (0,0): 1, (0,1): 1 },
        # r=4 (M5): C00=-1, C01=1
        { (0,0): -1, (1,0): 1 },
        # r=5 (M6): C11=1
        { (1,1): 1 },
        # r=6 (M7): C00=1
        { (0,0): 1 }
    ]
    
    # Iterate through blocks
    for i_start, i_end in I_parts:
        i_size = i_end - i_start
        for j_start, j_end in J_parts:
            j_size = j_end - j_start
            for k_start, k_end in K_parts:
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen Block
                    for r in range(7):
                        # U
                        for (di, dj), val in S_U[r].items():
                            i_glob = i_start + di
                            j_glob = j_start + dj
                            add_term(i_glob, j_glob, 0, val, 0, 0) # k is not used in U, but we need to add to the term list
                            # Wait, add_term expects i,j,k.
                            # U is indexed by i,j. The term is added to U[i*m+j, col].
                            # We need to associate this with a column.
                            # Let's collect coefficients per column.
                            pass
                        # This approach is messy. Let's just append to a list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)
                        # But we need to combine them per column.
                        # Actually, for a specific rank term r, we have specific non-zero entries.
                        # We can just add them to the terms list.
                        
                        # U part
                        for (di, dj), val in S_U[r].items():
                            i_glob = i_start + di
                            j_glob = j_start + dj
                            # We need to add (u_idx, 0, 0, val, 0, 0)? No.
                            # The term is a rank-1 tensor: u_vec * v_vec * w_vec.
                            # u_vec has non-zeros at specific (i,j).
                            # v_vec has non-zeros at specific (j,k).
                            # w_vec has non-zeros at specific (k,i).
                            
                            # We will store the vectors.
                            pass
                        
                    # Let's refactor: Store the 7 terms for this block.
                    block_terms = []
                    for r in range(7):
                        u_vec = {}
                        for (di, dj), val in S_U[r].items():
                            i_glob = i_start + di
                            j_glob = j_start + dj
                            u_vec[(i_glob, j_glob)] = val
                            
                        v_vec = {}
                        for (dj, dk), val in S_V[r].items():
                            j_glob = j_start + dj
                            k_glob = k_start + dk
                            v_vec[(j_glob, k_glob)] = val
                            
                        w_vec = {}
                        for (dk, di), val in S_W[r].items():
                            k_glob = k_start + dk
                            i_glob = i_start + di
                            w_vec[(k_glob, i_glob)] = val
                            
                        block_terms.append((u_vec, v_vec, w_vec))
                    
                    # Add to global terms
                    for u_vec, v_vec, w_vec in block_terms:
                        terms.append((u_vec, v_vec, w_vec))

                else:
                    # Schoolbook Block
                    for di in range(i_size):
                        for dj in range(j_size):
                            for dk in range(k_size):
                                i_glob = i_start + di
                                j_glob = j_start + dj
                                k_glob = k_start + dk
                                
                                u_vec = { (i_glob, j_glob): 1 }
                                v_vec = { (j_glob, k_glob): 1 }
                                w_vec = { (k_glob, i_glob): 1 }
                                terms.append((u_vec, v_vec, w_vec))

    # Now, we have a list of terms. Each term is a rank-1 tensor.
    # We need to construct U, V, W matrices.
    # The rank is the number of terms.
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for (i, j), val in u_vec.items():
            U[i * m + j][c] = val
        for (j, k), val in v_vec.items():
            V[j * p + k][c] = val
        for (k, i), val in w_vec.items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
