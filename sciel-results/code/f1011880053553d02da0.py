"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Total rank = 32 (for 1x4x8) + 56 (for 2x4x8 using Strassen) = 88
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: i=0 (1x4x8), Rank 32
    # Schoolbook
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 (2x4x8), Rank 56
    # Strassen decomposition
    # Local indices for i in {1,2} mapped to 0,1
    # A blocks (1x2):
    # A00: i=1, j=0,1
    # A01: i=1, j=2,3
    # A10: i=2, j=0,1
    # A11: i=2, j=2,3
    
    # B blocks (2x4):
    # B00: j=0,1, k=0..3
    # B01: j=0,1, k=4..7
    # B10: j=2,3, k=0..3
    # B11: j=2,3, k=4..7
    
    # Define Strassen terms
    # Each term t has:
    # X_coeffs: dict mapping (local_i, local_j) to coeff for X_t[j']
    # Actually, X_t is a 1x2 vector. X_t[0] and X_t[1].
    # X_t[j'] is linear comb of A elements.
    # Y_coeffs: dict mapping (local_j, local_k) to coeff for Y_t[j', k]
    # Y_t is 2x4 matrix.
    
    # We will define the linear combinations explicitly.
    
    # Helper to get global indices for A block elements
    def get_A_indices(block_id):
        # block_id 0: i=1, j=0,1
        # block_id 1: i=1, j=2,3
        # block_id 2: i=2, j=0,1
        # block_id 3: i=2, j=2,3
        i_glob = 1 + block_id // 2
        j_start = (block_id % 2) * 2
        return [(i_glob, j_start + dj) for dj in range(2)]
        
    def get_B_indices(block_id, j_local, k_local):
        # block_id 0: j=0,1, k=0..3
        # block_id 1: j=0,1, k=4..7
        # block_id 2: j=2,3, k=0..3
        # block_id 3: j=2,3, k=4..7
        j_glob = (block_id % 2) * 2 + j_local
        k_glob = (block_id // 2) * 4 + k_local
        return (j_glob, k_glob)

    # Strassen Definitions
    # t, X_blocks, Y_blocks
    # X_blocks: list of (block_id, coeff) for each j' in 0,1
    # Y_blocks: list of (block_id, coeff) for each (j', k) in 0..1, 0..3
    
    strassen_terms = []
    
    # M1: X = A00 + A11, Y = B00 + B11
    # X[0] = A00[0] + A11[0]
    # X[1] = A00[1] + A11[1]
    # Y[j', k] = B00[j', k] + B11[j', k]
    strassen_terms.append({
        'X': [[(0, 1.0), (3, 1.0)], [(0, 1.0), (3, 1.0)]],
        'Y': [[[(0, 1.0), (3, 1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M2: X = A10 + A11, Y = B00
    strassen_terms.append({
        'X': [[(2, 1.0), (3, 1.0)], [(2, 1.0), (3, 1.0)]],
        'Y': [[[(0, 1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M3: X = A00, Y = B01 - B11
    strassen_terms.append({
        'X': [[(0, 1.0)], [(0, 1.0)]],
        'Y': [[[(1, 1.0), (3, -1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M4: X = A11, Y = B10 - B00
    strassen_terms.append({
        'X': [[(3, 1.0)], [(3, 1.0)]],
        'Y': [[[(2, 1.0), (0, -1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M5: X = A00 + A01, Y = B11
    strassen_terms.append({
        'X': [[(0, 1.0), (1, 1.0)], [(0, 1.0), (1, 1.0)]],
        'Y': [[[(3, 1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M6: X = A10 - A00, Y = B00 + B01
    strassen_terms.append({
        'X': [[(2, 1.0), (0, -1.0)], [(2, 1.0), (0, -1.0)]],
        'Y': [[[(0, 1.0), (1, 1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # M7: X = A01 - A11, Y = B10 + B11
    strassen_terms.append({
        'X': [[(1, 1.0), (3, -1.0)], [(1, 1.0), (3, -1.0)]],
        'Y': [[[(2, 1.0), (3, 1.0)] for _ in range(4)] for _ in range(2)]
    })
    
    # Reconstruction Signs
    # C11 (i=1, k=0..3) = M1 + M4 - M5 + M7
    # C12 (i=1, k=4..7) = M3 + M5
    # C21 (i=2, k=0..3) = M2 + M4
    # C22 (i=2, k=4..7) = M1 - M2 + M3 + M6
    
    # Map t (0-6) and k (0-3) to list of (i_glob, k_glob, sign)
    # t=0 is M1, t=1 is M2, etc.
    
    def get_W_targets(t, k):
        targets = []
        # M1 (t=0)
        if t == 0:
            targets.append((1, k, 1.0))       # C11
            targets.append((2, 4 + k, 1.0))   # C22
        # M2 (t=1)
        elif t == 1:
            targets.append((2, k, 1.0))       # C21
            targets.append((2, 4 + k, -1.0))  # C22
        # M3 (t=2)
        elif t == 2:
            targets.append((1, 4 + k, 1.0))   # C12
            targets.append((2, 4 + k, 1.0))   # C22
        # M4 (t=3)
        elif t == 3:
            targets.append((1, k, 1.0))       # C11
            targets.append((2, k, 1.0))       # C21
        # M5 (t=4)
        elif t == 4:
            targets.append((1, k, -1.0))      # C11
            targets.append((1, 4 + k, 1.0))   # C12
        # M6 (t=5)
        elif t == 5:
            targets.append((2, 4 + k, 1.0))   # C22
        # M7 (t=6)
        elif t == 6:
            targets.append((1, k, 1.0))       # C11
            
        return targets

    for t in range(7):
        term_data = strassen_terms[t]
        X_coeffs = term_data['X']
        Y_coeffs = term_data['Y']
        
        for jp in range(2): # j_prime
            for k in range(4):
                # Get U coefficients
                u_coeffs = X_coeffs[jp]
                # Get V coefficients
                v_coeffs = Y_coeffs[jp][k]
                
                # Get W targets
                w_targets = get_W_targets(t, k)
                
                # If no targets, skip? (Should not happen)
                if not w_targets:
                    continue
                    
                # We need to add one rank-1 term for this (t, jp, k)
                # But wait, the rank-1 term in 1x2x4 is specific to (jp, k).
                # The U and V vectors are sparse.
                
                # U vector
                for block_id, coeff in u_coeffs:
                    indices = get_A_indices(block_id)
                    for (i_glob, j_glob) in indices:
                        U[i_glob * m + j_glob, col] += coeff
                        
                # V vector
                for block_id, coeff in v_coeffs:
                    # Y_coeffs[jp][k] is a list of (block_id, coeff)
                    # The block contributes to Y[jp, k]
                    # The element in the block is at (jp, k)
                    j_glob, k_glob = get_B_indices(block_id, jp, k)
                    V[j_glob * p + k_glob, col] += coeff
                    
                # W vector
                for (i_glob, k_glob, sign) in w_targets:
                    W[k_glob * n + i_glob, col] += sign
                    
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
