import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each tensor is (u, v, w) where u is 4-dim, v is 4-dim, w is 4-dim.
    Indices: u[i*m+j], v[j*p+k], w[k*n+i] with n=m=p=2.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]))
    # M2 = (A10+A11)B00
    terms.append(([0,0, 1,1], [1,0, 0,0], [0,1, 0,0]))
    # M3 = A00(B10-B11)
    terms.append(([1,0, 0,0], [0,0, 1,-1], [0,0, 1,0]))
    # M4 = A11(B00-B01)
    terms.append(([0,0, 0,1], [1,-1, 0,0], [0,0, 0,1]))
    # M5 = (A00+A01)B11
    terms.append(([1,1, 0,0], [0,0, 0,1], [0,0, 1,0]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([-1,0, 1,0], [1,1, 0,0], [0,1, 0,0]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0,1, 0,-1], [0,0, 1,1], [0,0, 0,1]))
    return terms

def _expand_strassen_block_4x4x4(block_idx, n=4):
    """
    Expand a single 2x2x2 Strassen term (index block_idx 0..6) into 
    4x4x4 rank-1 tensors.
    Returns list of 7 (u, v, w) vectors of length 16.
    """
    terms_2x2 = _strassen_2x2x2()
    u_local, v_local, w_local = terms_2x2[block_idx]
    
    # Define the mapping for each of the 7 Strassen products
    # Product 1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? 
    # No, Strassen computes:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The rank-1 tensors for the 2x2 matmul are (u_r, v_r, w_r).
    # The contribution to C is sum_r u_r^T A v_r w_r^T.
    # For the 4x4 case, we need to map the local indices to global.
    
    # The 7 terms correspond to specific combinations of A and B blocks.
    # We need to know which A blocks and B blocks are involved for each term.
    
    # Term 0 (M1): A blocks {00, 11}, B blocks {00, 11}
    # Term 1 (M2): A blocks {10, 11}, B blocks {00}
    # Term 2 (M3): A blocks {00}, B blocks {10, 11}
    # Term 3 (M4): A blocks {11}, B blocks {00, 01}
    # Term 4 (M5): A blocks {00, 01}, B blocks {11}
    # Term 5 (M6): A blocks {10, 00}, B blocks {00, 01}
    # Term 6 (M7): A blocks {01, 11}, B blocks {10, 11}
    
    # Coefficients for A blocks:
    a_blocks = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,1)],      # M2
        [(0,0,1)],               # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (0,0,-1)],     # M6
        [(0,1,1), (1,1,-1)]      # M7
    ]
    
    # Coefficients for B blocks:
    b_blocks = [
        [(0,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,0,1), (1,1,-1)],     # M3
        [(0,0,1), (0,1,-1)],     # M4
        [(1,1,1)],               # M5
        [(0,0,1), (0,1,1)],      # M6
        [(1,0,1), (1,1,1)]       # M7
    ]
    
    # Coefficients for C blocks (output):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_coeffs = [
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], # M1 adds to all
        [(1,0,1), (1,1,1)],                   # M2 adds to C10, C11 (with -1 for C11? No, M2 is +C10, -C11? 
        # Let's check Strassen formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M1: C00(+), C11(+). C01(0), C10(0).
        # M2: C10(+), C11(-).
        # M3: C01(+), C11(+).
        # M4: C00(+), C10(+).
        # M5: C00(-), C01(+).
        # M6: C11(+).
        # M7: C00(+).
        
        [(0,0,1), (1,1,1)],                   # M1
        [(1,0,1), (1,1,-1)],                  # M2
        [(0,1,1), (1,1,1)],                   # M3
        [(0,0,1), (1,0,1)],                   # M4
        [(0,0,-1), (0,1,1)],                  # M5
        [(1,1,1)],                            # M6
        [(0,0,1)]                             # M7
    ]
    
    expanded_terms = []
    
    for r in range(7):
        u_global = np.zeros(16)
        v_global = np.zeros(16)
        w_global = np.zeros(16)
        
        # Expand u
        for (bi, bj, coeff) in a_blocks[block_idx]:
            for di in range(2):
                for dj in range(2):
                    gi = bi * 2 + di
                    gj = bj * 2 + dj
                    u_global[gi * 4 + gj] += coeff * u_local[di * 2 + dj]
                    
        # Expand v
        for (bi, bj, coeff) in b_blocks[block_idx]:
            for di in range(2):
                for dj in range(2):
                    gi = bi * 2 + di
                    gj = bj * 2 + dj
                    v_global[gi * 4 + gj] += coeff * v_local[di * 2 + dj]
                    
        # Expand w
        for (bi, bj, coeff) in c_coeffs[block_idx]:
            for di in range(2):
                for dj in range(2):
                    gi = bi * 2 + di
                    gj = bj * 2 + dj
                    w_global[gi * 4 + gj] += coeff * w_local[di * 2 + dj]
                    
        expanded_terms.append((u_global, v_global, w_global))
        
    return expanded_terms

def _construct_4x4x4_strassen(n=4):
    """Construct rank-49 decomposition for 4x4x4 matmul."""
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    for block_idx in range(7):
        terms = _expand_strassen_block_4x4x4(block_idx, n)
        for r in range(7):
            u, v, w = terms[r]
            U[:, col] = u
            V[:, col] = v
            W[:, col] = w
            col += 1
            
    return U, V, W

def _construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 Strassen (rank 49) for k=0..3
    U1, V1, W1 = _construct_4x4x4_strassen()
    
    # Part 2: 4x4x1 Schoolbook (rank 16) for k=4
    U2 = np.zeros((16, 16))
    V2 = np.zeros((16, 16))
    W2 = np.zeros((16, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U2[i * 4 + j, col] = 1
            V2[j * 4 + 4, col] = 1 # k=4
            W2[4 * 4 + i, col] = 1 # k=4, i
            col += 1
            
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
