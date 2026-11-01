import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm parameters for 2x2 blocks
    # Blocks are defined by column offsets in B and C
    # B is 2x6, so 3 blocks of 2x2
    # C is 2x6, so 3 blocks of 2x2
    
    # Define the 7 Strassen terms for a single block
    # Each term is a tuple (u_indices, u_values, v_indices, v_values, w_indices, w_values)
    # Indices are local to the block for V and W, we will map them to global.
    # U indices are global (i, j) as A is shared.
    
    # Local V indices: (j, k_local) where k_local is 0 or 1
    # Local W indices: (k_local, i)
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1  -> C00 += P1, C11 += P1
    t1 = (
        [(0, 0), (1, 1)], [1, 1],
        [(0, 0), (1, 1)], [1, 1],
        [(0, 0), (1, 1)], [1, 1]
    )
    
    # Term 2: (A10 + A11) * B00
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1, (1,1)=-1 -> C10 += P2, C11 -= P2
    t2 = (
        [(1, 0), (1, 1)], [1, 1],
        [(0, 0)], [1],
        [(0, 1), (1, 1)], [1, -1]
    )
    
    # Term 3: A00 * (B01 - B11)
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (1,0)=1, (1,1)=1 -> C01 += P3, C11 += P3
    t3 = (
        [(0, 0)], [1],
        [(0, 1), (1, 1)], [1, -1],
        [(1, 0), (1, 1)], [1, 1]
    )
    
    # Term 4: A11 * (B10 - B00)
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (0,0)=1, (0,1)=1 -> C00 += P4, C10 += P4
    t4 = (
        [(1, 1)], [1],
        [(1, 0), (0, 0)], [1, -1],
        [(0, 0), (0, 1)], [1, 1]
    )
    
    # Term 5: (A00 + A01) * B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (1,0)=1, (0,0)=-1 -> C01 += P5, C00 -= P5
    t5 = (
        [(0, 0), (0, 1)], [1, 1],
        [(1, 1)], [1],
        [(1, 0), (0, 0)], [1, -1]
    )
    
    # Term 6: (A10 - A00) * (B00 + B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1 -> C11 += P6
    t6 = (
        [(1, 0), (0, 0)], [1, -1],
        [(0, 0), (0, 1)], [1, 1],
        [(1, 1)], [1]
    )
    
    # Term 7: (A01 - A11) * (B10 + B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1 -> C00 += P7
    t7 = (
        [(0, 1), (1, 1)], [1, -1],
        [(1, 0), (1, 1)], [1, 1],
        [(0, 0)], [1]
    )
    
    terms = [t1, t2, t3, t4, t5, t6, t7]
    
    for b in range(3):
        col_offset = b * 7
        for t_idx, term in enumerate(terms):
            r = col_offset + t_idx
            u_indices, u_values, v_indices, v_values, w_indices, w_values = term
            
            # Fill U
            # Global index: i * m + j
            for idx, val in zip(u_indices, u_values):
                i, j = idx
                U[i * m + j][r] = val
                
            # Fill V
            # Global index: j * p + k
            # k = 2 * b + k_local
            for idx, val in zip(v_indices, v_values):
                j, k_local = idx
                k_global = 2 * b + k_local
                V[j * p + k_global][r] = val
                
            # Fill W
            # Global index: k * n + i
            # k = 2 * b + k_local
            for idx, val in zip(w_indices, w_values):
                k_local, i = idx
                k_global = 2 * b + k_local
                W[k_global * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
