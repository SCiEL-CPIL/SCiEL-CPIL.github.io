"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # We will build the decomposition column by column.
    # Each column corresponds to one multiplication term.
    # U, V, W are lists of lists (rows x cols) initially, then converted to arrays.
    # Actually, it's easier to build arrays of 0s and fill them.
    
    # We don't know the rank in advance easily, so we'll use a list of tuples (u_val, v_val, w_val)
    # where u_val is the index in U (i*m+j), v_val in V (j*p+k), w_val in W (k*n+i)
    # and the value is the coefficient (which will be 1, -1, etc for Strassen).
    # Wait, the standard CP decomposition has U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # If we use Strassen, the "multiplication" is not just A[i,j]*B[j,k].
    # Strassen computes C[i,k] using linear combinations of A and B.
    # The tensor T[i,j,k] = 1 represents the contribution of A[i,j]*B[j,k] to C[i,k].
    # A rank-r decomposition T = sum_c u_c v_c w_c means:
    # sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
    # Here a=i*m+j, b=j*p+k, d=k*n+i.
    
    # Strassen for 2x2:
    # C[0,0] = M1 - M2 + M4 - M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    # where
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # This structure implies that the "tensor" for Strassen is a sum of 7 rank-1 terms,
    # but the vectors U, V, W contain the linear combinations.
    
    # Let's define the Strassen decomposition for a specific 2x2x2 block.
    # The block is defined by row indices i1, i2, col indices k1, k2, and mid index j.
    # Actually, matrix multiplication is sum_j A[i,j] B[j,k].
    # Strassen applies to the 2x2 matrix multiplication where the mid dimension is also 2?
    # No, standard Strassen is for 2x2 * 2x2 = 2x2. The mid dimension is 2.
    # Here our mid dimension is m=5.
    # So we can only apply Strassen if we group the mid dimension into pairs.
    # For m=5, we can have two 2x2 blocks (using j=0,1 and j=2,3) and one 1x1 block (j=4).
    # And we need to handle the row/col dimensions.
    # n=3, p=6.
    # Rows: 0,1,2. We can form one 2x2 block (rows 0,1) and one 1x1 (row 2).
    # Cols: 0..5. We can form three 2x2 blocks (cols 0,1; 2,3; 4,5).
    
    # So the strategy is:
    # 1. Iterate over all combinations of "Row Block", "Col Block", and "Mid Block".
    # 2. For 2x2 Row Block, 2x2 Col Block, 2x2 Mid Block: Use Strassen (rank 7).
    # 3. For other combinations (involving 1x1 blocks): Use Schoolbook (rank 4, 2, or 1).
    
    # Let's define the blocks:
    # Row blocks: [(0,1), (2,2)]  -> sizes 2, 1
    # Mid blocks: [(0,1), (2,3), (4,4)] -> sizes 2, 2, 1
    # Col blocks: [(0,1), (2,3), (4,5)] -> sizes 2, 2, 2
    
    # We will accumulate the contributions.
    # Each "term" in the decomposition is a rank-1 tensor u ⊗ v ⊗ w.
    # For Schoolbook: u = e_{i*m+j}, v = e_{j*p+k}, w = e_{k*n+i}.
    # For Strassen: The "u" vector is a linear combination of e_{i*m+j} for i in row_block, j in mid_block.
    # The "v" vector is a linear combination of e_{j*p+k} for j in mid_block, k in col_block.
    # The "w" vector is a linear combination of e_{k*n+i} for k in col_block, i in row_block.
    
    # We will store the final U, V, W as numpy arrays.
    # We don't know the total rank yet, so we'll use a list of (u_vec, v_vec, w_vec) tuples.
    # u_vec, v_vec, w_vec are dictionaries or sparse representations.
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        # u_dict: {index: coeff}
        # v_dict: {index: coeff}
        # w_dict: {index: coeff}
        terms.append((u_dict, v_dict, w_dict))
        
    # Strassen coefficients for the 7 products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's define the Strassen decomposition for a generic 2x2x2 block.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1} relative to the block.
    # We map these to global indices.
    
    def add_strassen_block(row_idx, col_idx, mid_idx):
        # row_idx: (i1, i2)
        # col_idx: (k1, k2)
        # mid_idx: (j1, j2)
        i1, i2 = row_idx
        k1, k2 = col_idx
        j1, j2 = mid_idx
        
        # M1
        u = {i1*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M2
        u = {i2*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M3
        u = {i1*m+j1: 1}
        v = {j1*p+k2: 1, j2*p+k2: -1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M4
        u = {i2*m+j2: 1}
        v = {j2*p+k1: 1, j1*p+k1: -1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M5
        u = {i1*m+j1: 1, i1*m+j2: 1}
        v = {j2*p+k2: 1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M6
        u = {i2*m+j1: 1, i1*m+j1: -1}
        v = {j1*p+k1: 1, j1*p+k2: 1}
        w = {k1*n+i2: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M7
        u = {i1*m+j2: 1, i2*m+j2: -1}
        v = {j2*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k2*n+i1: 1}
        add_term(u, v, w)
        
    def add_schoolbook_block(row_idx, col_idx, mid_idx):
        # row_idx: (i1, i2)
        # col_idx: (k1, k2)
        # mid_idx: (j1, j2)
        for i in row_idx:
            for j in mid_idx:
                for k in col_idx:
                    u = {i*m+j: 1}
                    v = {j*p+k: 1}
                    w = {k*n+i: 1}
                    add_term(u, v, w)
    
    # Define blocks
    row_blocks = [(0, 1), (2, 2)]
    mid_blocks = [(0, 1), (2, 3), (4, 4)]
    col_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for rb in row_blocks:
        for mb in mid_blocks:
            for cb in col_blocks:
                # If all are size 2, use Strassen
                if len(rb) == 2 and len(mb) == 2 and len(cb) == 2:
                    add_strassen_block(rb, cb, mb)
                else:
                    add_schoolbook_block(rb, cb, mb)
    
    # Now convert terms to U, V, W matrices
    # Find max indices
    max_u = max(max(u.keys()) for u, v, w in terms)
    max_v = max(max(v.keys()) for u, v, w in terms)
    max_w = max(max(w.keys()) for u, v, w in terms)
    
    rank = len(terms)
    
    U = np.zeros((max_u + 1, rank))
    V = np.zeros((max_v + 1, rank))
    W = np.zeros((max_w + 1, rank))
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx, c] = val
        for idx, val in v.items():
            V[idx, c] = val
        for idx, val in w.items():
            W[idx, c] = val
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Ensure shapes are correct
        # U should be (n*m, rank)
        # V should be (m*p, rank)
        # W should be (p*n, rank)
        # Our max_u is i*m+j, max index is (n-1)*m + (m-1) = n*m - 1. So size is n*m.
        # Our max_v is j*p+k, max index is (m-1)*p + (p-1) = m*p - 1. So size is m*p.
        # Our max_w is k*n+i, max index is (p-1)*n + (n-1) = p*n - 1. So size is p*n.
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
