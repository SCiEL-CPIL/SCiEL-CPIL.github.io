import numpy as np
import time

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix.
    
    The problem is equivalent to computing 7 independent 2x2 matrix products.
    The tensor for 2x2x2 matrix multiplication has rank 7 (Strassen's algorithm).
    Since the 14 columns of the second matrix can be grouped into 7 disjoint
    2-column blocks, the total tensor is the direct sum of 7 copies of the
    2x2x2 multiplication tensor.
    
    The rank of a direct sum of tensors is the sum of their ranks.
    Therefore, the minimal rank is 7 * 7 = 49.
    
    This program returns a decomposition of rank 49 using Strassen's algorithm
    applied to each of the 7 blocks.
    """
    n, m, p = 2, 2, 14
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_blocks = p // 2  # 7 blocks of width 2
    rank = num_blocks * 7  # 49

    # Strassen's 7 terms for 2x2 * 2x2
    # Each term is a tuple of three lists: (A_indices, B_indices, C_indices)
    # Indices are local to the 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # The sign is embedded in the value.
    # We store them as (idx, val) pairs.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the terms as:
    # (list of (a_idx, a_val), list of (b_idx, b_val), list of (c_idx, c_val))
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # Wait, C10 gets +M2, C11 gets -M2.
        # Let's re-verify the standard Strassen assignment to C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the contribution of each M_r to C_{i,k}.
        # M1: C00 (+1), C11 (+1) -> c_idx 0:+1, 3:+1
        # M2: C10 (+1), C11 (-1) -> c_idx 2:+1, 3:-1
        # M3: C01 (+1), C11 (+1) -> c_idx 1:+1, 3:+1
        # M4: C00 (+1), C10 (+1) -> c_idx 0:+1, 2:+1
        # M5: C00 (-1), C01 (+1) -> c_idx 0:-1, 1:+1
        # M6: C11 (+1)             -> c_idx 3:+1
        # M7: C00 (+1)             -> c_idx 0:+1
        
        # A parts:
        # M1: A00+A11 -> 0:+1, 3:+1
        # M2: A10+A11 -> 2:+1, 3:+1
        # M3: A00     -> 0:+1
        # M4: A11     -> 3:+1
        # M5: A00+A01 -> 0:+1, 1:+1
        # M6: A10-A00 -> 2:+1, 0:-1
        # M7: A01-A11 -> 1:+1, 3:-1
        
        # B parts:
        # M1: B00+B11 -> 0:+1, 3:+1
        # M2: B00     -> 0:+1
        # M3: B01-B11 -> 1:+1, 3:-1
        # M4: B10-B00 -> 2:+1, 0:-1
        # M5: B11     -> 3:+1
        # M6: B00+B01 -> 0:+1, 1:+1
        # M7: B10+B11 -> 2:+1, 3:+1
    ]
    
    # Constructing the list based on the above derivation
    terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_terms, b_terms, c_terms = terms[s]

            # Fill U
            for idx, val in a_terms:
                i = idx // 2
                j = idx % 2
                U[i * m + j, col] += val

            # Fill V
            for idx, val in b_terms:
                j = idx // 2
                k_local = idx % 2
                k_global = 2 * t + k_local
                V[j * p + k_global, col] += val

            # Fill W
            for idx, val in c_terms:
                i = idx // 2
                k_local = idx % 2
                k_global = 2 * t + k_local
                W[k_global * n + i, col] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
