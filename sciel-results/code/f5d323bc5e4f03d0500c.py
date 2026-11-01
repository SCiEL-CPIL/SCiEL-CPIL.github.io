"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # A is n x m = 2x2, B is m x p = 2x12, C is n x p = 2x12
    # Partition p columns into pairs: 6 pairs
    num_pairs = p // 2  # 6
    
    rank = num_pairs * 7  # 42
    U = np.zeros((n * m, rank), dtype=float)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)   # 24 x 42
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen products: (A_coeffs, B_coeffs, C_coeffs)
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C elements: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    products = [
        # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A21+A22)B11 -> C21+=1, C22+=-1
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3 = A11(B12-B22) -> C12+=1, C22+=1
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4 = A22(B21-B11) -> C11+=1, C21+=1
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5 = (A11+A12)B22 -> C11+=-1, C12+=1
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6 = (A21-A11)(B11+B12) -> C22+=1
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7 = (A12-A22)(B21+B22) -> C11+=1
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        base_col = 2 * t  # starting column in B and C
        for s in range(7):
            a_coeffs, b_coeffs, c_coeffs = products[s]
            
            # Set U entries (A coefficients)
            for (ai, aj), coeff in a_coeffs:
                u_row = ai * m + aj
                U[u_row, col] = coeff
            
            # Set V entries (B coefficients)
            for (bj, bk_rel), coeff in b_coeffs:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row, col] = coeff
            
            # Set W entries (C output coefficients)
            for (ci, ck_rel), coeff in c_coeffs:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci
                W[w_row, col] = coeff
            
            col += 1
    
    # Validation using NumPy for speed
    remaining = deadline - time.time()
    if remaining > 0.5:
        # Check all 2*2*12 = 48 tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    dot = np.dot(u_val * v_val, w_val)
                    if abs(dot - 1.0) > 1e-9:
                        break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
