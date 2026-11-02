import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 6
    rank = 21
    
    # Initialize U, V, W as lists of lists
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen constants for 2x2 multiplication
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B local indices: 0:B00, 1:B01, 2:B10, 3:B11
    # C local indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # The 7 Strassen multiplications
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B10 + B11)
    # M7 = (A01 - A11)(B00 + B01)
    
    # U coefficients (for A)
    # Each is a dict {index: value}
    u_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    # V coefficients (for B)
    # Each is a list of (local_idx, value)
    v_coeffs = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)]
    ]
    
    # W coefficients (for C)
    # Each is a list of (local_idx, value)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    w_coeffs = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)],  # M3
        [(0, 1), (2, 1)],  # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)],          # M6
        [(0, 1)]           # M7
    ]
    
    # Fill the matrices for 3 blocks of columns in B
    # Block b corresponds to columns 2*b and 2*b+1 of B
    # Global B index: row j (0..1), col k (0..5) -> idx = j*6 + k
    # Global C index: row i (0..1), col k (0..5) -> idx = i*6 + k? 
    # Wait, the trace convention for W[k*n+i, r] is specified in the task?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W is the coefficient for C.
    # C is 2x6. Indices (i, k). 
    # If W is flattened, how is it ordered?
    # Task says: W[k*n+i, r]. 
    # This implies index = k*n + i.
    # Here n=2, so index = 2*k + i.
    # k is the column index of C (0..5), i is the row index (0..1).
    
    for b in range(3):
        for t in range(7):
            r_idx = b * 7 + t
            
            # U is independent of b, but we write it 3 times? No, U is shared.
            # But the tensor decomposition requires U, V, W to be consistent.
            # Since the blocks are independent in B and C, the terms for block 0, 1, 2 are distinct.
            # U is the same for all blocks.
            # V and W depend on the block.
            
            for a_idx, val in u_coeffs[t].items():
                U[a_idx][r_idx] = val
                
            for v_local, val in v_coeffs[t]:
                # v_local is 0..3 corresponding to B00, B01, B10, B11
                # Map to global col: 2*b + (v_local % 2)
                # row: v_local // 2
                j = v_local // 2
                k_local = v_local % 2
                k_global = 2 * b + k_local
                v_flat_idx = j * p + k_global
                V[v_flat_idx][r_idx] = val
                
            for w_local, val in w_coeffs[t]:
                # w_local is 0..3 corresponding to C00, C01, C10, C11
                # Map to global col: 2*b + (w_local % 2)
                # row: w_local // 2
                i = w_local // 2
                k_local = w_local % 2
                k_global = 2 * b + k_local
                # Convention W[k*n + i, r]
                w_flat_idx = k_global * n + i
                W[w_flat_idx][r_idx] = val
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
