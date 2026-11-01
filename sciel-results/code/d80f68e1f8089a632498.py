import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Returns a rank-35 exact tensor decomposition for 2x2 * 2x10 matmul.
    
    Strategy:
    The tensor T_{2,2,10} decomposes into 5 independent T_{2,2,2} blocks 
    corresponding to pairs of columns in the B matrix.
    Each T_{2,2,2} block has a minimal rank of 7 (Strassen's algorithm).
    Since the A-matrix inputs are shared but the B and C outputs are disjoint 
    across blocks, and the linear forms on A in the minimal decomposition 
    are independent, the total rank is 5 * 7 = 35.
    
    This implementation constructs the Strassen decomposition for each block 
    and concatenates them.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]] -> indices 0,1,2,3 in U
    # B = [[e, f], [g, h]] -> indices 0,1,2,3 in local V
    # C = [[c00, c01], [c10, c11]] -> indices 0,1,2,3 in local W (flattened k*n+i)
    #   c00 -> 0, c01 -> 1, c10 -> 2, c11 -> 3
    
    # Terms: (U_indices, U_vals, V_local_indices, V_vals, W_local_indices, W_vals)
    # We define the 7 rank-1 tensors for one 2x2x2 block.
    
    strassen_terms = [
        # M1: (a+d)(e+h) -> C00(+), C11(+)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2: (c+d)e -> C10(+), C11(-)
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3: a(f-h) -> C01(+), C11(+)
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4: d(g-e) -> C00(+), C10(+)
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5: (a+b)h -> C00(-), C01(+)
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6: (c-a)(e+f) -> C11(+)
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7: (b-d)(g+h) -> C00(+)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])
    ]
    
    for k in range(num_blocks):
        if time.time() > deadline:
            break
            
        base_rank = k * 7
        
        # Mapping for V (B matrix)
        # B is 2x10. Flattened row-major: B[i, j] -> index i*10 + j
        # Block k uses columns 2k and 2k+1
        # Local V indices: 0:e(0,2k), 1:f(0,2k+1), 2:g(1,2k), 3:h(1,2k+1)
        v_map = {
            0: 0 * p + (2 * k),
            1: 0 * p + (2 * k + 1),
            2: 1 * p + (2 * k),
            3: 1 * p + (2 * k + 1)
        }
        
        # Mapping for W (C matrix)
        # C is 2x10. Flattened row-major: C[i, j] -> index i*10 + j
        # Wait, the task says "official trace convention W[k*n+i,r]"
        # This implies W is indexed by (k, i) where k is row of C? or column?
        # Standard matmul: C = A B. A is 2x2, B is 2x10. C is 2x10.
        # C[r, c] = sum_k A[r, k] B[k, c].
        # The tensor indices are (A_idx, B_idx, C_idx).
        # U corresponds to A. Size n*m = 4.
        # V corresponds to B. Size m*p = 20.
        # W corresponds to C. Size p*n = 20.
        
        # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
        # This usually means the index of W is `k*n + i`.
        # In the context of C (2x10), if we flatten it, how?
        # If W[k*n+i] corresponds to C element, and n=2.
        # If k is the column index of C (0..9) and i is the row index of C (0..1)?
        # Then index = col * 2 + row.
        # This is column-major flattening of C.
        # Let's verify with the seed program's logic.
        # Seed: w_map 0: 4k, 1: 4k+2, 2: 4k+1, 3: 4k+3.
        # Local 0: C00 (row 0, col 2k). Index 4k.
        # Local 1: C01 (row 0, col 2k+1). Index 4k+2.
        # Local 2: C10 (row 1, col 2k). Index 4k+1.
        # Local 3: C11 (row 1, col 2k+1). Index 4k+3.
        
        # Let's check the formula index = col * n + row.
        # C00: col=2k, row=0. Index = 2k*2 + 0 = 4k. Matches.
        # C01: col=2k+1, row=0. Index = (2k+1)*2 + 0 = 4k+2. Matches.
        # C10: col=2k, row=1. Index = 2k*2 + 1 = 4k+1. Matches.
        # C11: col=2k+1, row=1. Index = (2k+1)*2 + 1 = 4k+3. Matches.
        
        # So W is flattened column-major (or index = col*n + row).
        
        w_map = {
            0: (2 * k) * n + 0,
            1: (2 * k + 1) * n + 0,
            2: (2 * k) * n + 1,
            3: (2 * k + 1) * n + 1
        }
        
        for r, term in enumerate(strassen_terms):
            global_rank = base_rank + r
            
            u_indices, u_vals, v_indices, v_vals, w_indices, w_vals = term
            
            for u_idx, val in zip(u_indices, u_vals):
                U[u_idx, global_rank] = val
                
            for v_local_idx, val in zip(v_indices, v_vals):
                v_global_idx = v_map[v_local_idx]
                V[v_global_idx, global_rank] = val
                
            for w_local_idx, val in zip(w_indices, w_vals):
                w_global_idx = w_map[w_local_idx]
                W[w_global_idx, global_rank] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
