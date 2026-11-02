"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    
    Strategy:
    - Split B (4x6) into B1 (4x4) and B2 (4x2)
    - C1 = A * B1 (4x4x4): Use Strassen tensor Strassen, rank 49
    - C2 = A * B2 (4x4x2): Use 4 Strassen 2x2x2 products, rank 28
    - Total rank: 49 + 28 = 77
    
    The 4x4x4 part:
    Partition A = [[A11, A12], [A21, A22]], B1 = [[B11, B12], [B21, B22]]
    C1 = [[A11*B11+A12*B21, A11*B12+A12*B22],
          [A21*B11+A22*B21, A21*B12+A22*B22]]
    Each 2x2x2 product uses Strassen (rank 7).
    Using tensor product: 4x4x4 = (2x2x2) x (2x2x2) via Kronecker, rank 7*7=49.
    
    The 4x4x2 part:
    Partition A = [[A11, A12], [A21, A22]], B2 = [[B1], [B2]] (each 2x2)
    C2 = [[A11*B1 + A12*B2],
          [A21*B1 + A22*B2]]
    4 independent 2x2x2 products, each Strassen rank 7, total 28.
    """
    # Strassen 2x2x2 decomposition (rank 7)
    # C = A * B where A, B, C are 2x2
    # Terms: (U, V, W) where U is 2x2, V is 2x2, W is 2x2
    # C[i,j] = sum_r U[i,j,r] * V[?,r] * W[?,r]
    # Actually: alpha_r = sum(A * U_r), beta_r = sum(B * V_r), C += alpha_r * beta_r * W_r
    
    strassen_2x2 = [
        # r=0: m1 = (a11+a22)(b11+b22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # r=1: m2 = (a21+a22)*b11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}),
        # r=2: m3 = a11*(b21-b11)
        ({(0,0): 1}, {(0,1): 1, (1,0): -1, (1,1): -1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # r=3: m4 = a22*(b12-b22)
        ({(1,1): 1}, {(0,0): 1, (0,1): -1, (1,0): 1, (1,1): -1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # r=4: m5 = (a11+a12)*b22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): -1, (1,0): 1, (1,1): -1}),
        # r=5: m6 = (a21-a11)*(b11+b12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (1,0): 1, (1,1): 1}),
        # r=7: m7 = (a12-a22)*(b21+b22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}),
    ]
    
    rank_terms = []
    
    # Part 1: 4x4x4 via Kronecker product of two Strassen decompositions
    # A_4x4 partitioned into 2x2 blocks, B1_4x4 partitioned into 2x2 blocks
    # C1_4x4 partitioned into 2x2 blocks
    # Use tensor product: for each pair of Strassen terms (s1, s2), create one term
    for s1 in range(7):
        u1, v1, w1 = strassen_2x2[s1]
        for s2 in range(7):
            u2, v2, w2 = strassen_2x2[s2]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: 4x4, indexed by (block_row, block_col, inner_row, inner_col)
            # Global index: (2*br + ir, 2*bc + ic)
            for (br, bc), c1 in u1.items():
                for (ir, ic), c2 in u2.items():
                    gi = 2 * br + ir
                    gj = 2 * bc + ic
                    coeff = c1 * c2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V: 4x4 (for B1 which is 4x4), indexed by (block_row, block_col, inner_row, inner_col)
            # Global index: (2*br + ir, 2*bc + ic) where br is row-block, bc is col-block
            for (br, bc), c1 in v1.items():
                for (ir, ic), c2 in v2.items():
                    gi = 2 * br + ir
                    gj = 2 * bc + ic
                    coeff = c1 * c2
                    key = (gi, gj)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W: 4x4 (for C1 which is 4x4), but we need to map to global C index
            # C1 occupies columns 0-3 of the full C (4x6)
            # W is indexed by (row, col) in the 4x4 C1 block
            for (ir, ic), c1 in w1.items():
                for (br, bc), c2 in w2.items():
                    gi = 2 * ir + br
                    gj = 2 * ic + bc
                    coeff = c1 * c2
                    key = (gj, gi)  # W[k*n + i] where k is col, i is row
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    # Part 2: 4x4x2 via 4 Strassen 2x2x2 products
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B2 = [[B1], [B2]] (each 2x2, B2 is 4x2)
    # C2 = [[A11*B1 + A12*B2],
    #       [A21*B1 + A22*B2]]
    # C2 is 4x2, occupying columns 4-5 of full C
    
    # Product 1: A11 * B1 -> C2[0:2, 0:2]
    # Product 2: A12 * B2 -> C2[0:2, 0:2]
    # Product 3: A21 * B1 -> C2[2:4, 0:2]
    # Product 4: A22 * B2 -> C2[2:4, 0:2]
    
    # For each 2x2x2 Strassen product, we need to map to global indices
    # A is 4x4: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B2 is 4x2: B1 = B2[0:2, 0:2] = B[0:2, 4:6], B2 = B2[2:4, 0:2] = B[2:4, 4:6]
    # C2 is 4x2: C2[0:2, 0:2] = C[0:2, 4:6], C2[2:4, 0:2] = C[2:4, 4:6]
    
    # Define the 4 products:
    # (A_block, B_block, C_block) where each is specified by (row_start, col_start) for 2x2 blocks
    products_4x4x2 = [
        # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
        (0, 0, 0, 4, 0, 4),  # A11 * B1 -> C2[0:2, 0:2]
        (0, 2, 2, 4, 0, 4),  # A12 * B2 -> C2[0:2, 0:2]
        (2, 0, 0, 4, 2, 4),  # A21 * B1 -> C2[2:4, 0:2]
        (2, 2, 2, 4, 2, 4),  # A22 * B2 -> C2[2:4, 0:2]
    ]
    
    for (ar, ac, br, bc, cr, cc) in products_4x4x2:
        for s in range(7):
            u_s, v_s, w_s = strassen_2x2[s]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i, j), cu in u_s.items():
                gi = ar + i
                gj = ac + j
                key = (gi, gj)
                U_global[key] = U_global.get(key, 0) + cu
            
            for (i, j), cv in v_s.items():
                gi = br + i
                gj = bc + j
                key = (gi, gj)
                V_global[key] = V_global.get(key, 0) + cv
            
            for (i, j), cw in w_s.items():
                gi = cr + i
                gj = cc + j
                key = (gj, gi)  # W[k*n + i]
                W_global[key] = W_global.get(key, 0) + cw
            
            rank_terms.append((U_global, V_global, W_global))
    
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (i, j), coeff in v_dict.items():
            V[i * p + j, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Check deadline before verification
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
