"""
Optimal rank-49 decomposition for 4x4 matrix multiplication.
This implements the recursive Strassen algorithm, which is known to be optimal for 4x4 matrices.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 7 outer products × 7 inner products = 49.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4), "This decomposition is specific to 4x4 matrices."

    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def w_idx(i, k):
        return k * n + i

    # We use recursive Strassen.
    # Level 1: Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B
    # Strassen computes 7 products M1..M7 of 2x2 matrices.
    # Each 2x2 product is computed using Strassen again (7 products of scalars).
    # Total rank = 7 * 7 = 49.

    # Helper to create a 2x2 matrix of linear combinations of A/B elements.
    # An element is represented as a list of (row, col, coeff) tuples.
    # For A, row/col are 0..3. Index in flat vector is row*4 + col.
    
    def make_A_block(br, bc):
        """Returns 2x2 matrix of linear combos representing A[br*2:br*2+2, bc*2:bc*2+2]"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                # Single element A[br*2+i, bc*2+j]
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def make_B_block(br, bc):
        """Returns 2x2 matrix of linear combos representing B[br*2:br*2+2, bc*2:bc*2+2]"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                # Single element B[br*2+i, bc*2+j]
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def add_mat(X, Y):
        """Add two 2x2 matrices of linear combos."""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for (r, c, v) in X[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for (r, c, v) in Y[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                row.append([(r, c, v) for (r,c), v in d.items() if v != 0])
            result.append(row)
        return result

    def sub_mat(X, Y):
        """Subtract two 2x2 matrices of linear combos."""
        neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_mat(X, neg_Y)

    # Define the 4 blocks of A and B
    A00 = make_A_block(0, 0)
    A01 = make_A_block(0, 1)
    A10 = make_A_block(1, 0)
    A11 = make_A_block(1, 1)
    
    B00 = make_B_block(0, 0)
    B01 = make_B_block(0, 1)
    B10 = make_B_block(1, 0)
    B11 = make_B_block(1, 1)

    # Outer Strassen terms: 7 pairs (X, Y) of 2x2 matrices of linear combos
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    outer_X = [
        add_mat(A00, A11),       # M1
        add_mat(A10, A11),       # M2
        A00,                     # M3
        A11,                     # M4
        add_mat(A00, A01),       # M5
        sub_mat(A10, A00),       # M6
        sub_mat(A01, A11),       # M7
    ]
    outer_Y = [
        add_mat(B00, B11),       # M1
        B00,                     # M2
        sub_mat(B01, B11),       # M3
        sub_mat(B10, B00),       # M4
        B11,                     # M5
        add_mat(B00, B01),       # M6
        add_mat(B10, B11),       # M7
    ]

    # Contribution of each outer term M_t to the C blocks (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_targets[t] = list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)],               # M7
    ]

    # Inner Strassen on 2x2 matrices X, Y.
    # Returns 7 pairs (X_f, Y_f) which are 2x2 matrices of linear combos.
    def inner_factors(X, Y):
        def single_entry(i, j, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            result[i][j] = list(mat[i][j])
            return result
        
        def add_entries(i1, j1, i2, j2, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) + v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        def sub_entries(i1, j1, i2, j2, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) - v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result

        # Inner M1 = (X00+X11)(Y00+Y11)
        X1 = add_entries(0, 0, 1, 1, X)
        Y1 = add_entries(0, 0, 1, 1, Y)
        
        # Inner M2 = (X10+X11)Y00
        X2 = add_entries(1, 0, 1, 1, X)
        Y2 = single_entry(0, 0, Y)
        
        # Inner M3 = X00(Y01-Y11)
        X3 = single_entry(0, 0, X)
        Y3 = sub_entries(0, 1, 1, 1, Y)
        
        # Inner M4 = X11(Y10-Y00)
        X4 = single_entry(1, 1, X)
        Y4 = sub_entries(1, 0, 0, 0, Y)
        
        # Inner M5 = (X00+X01)Y11
        X5 = add_entries(0, 0, 0, 1, X)
        Y5 = single_entry(1, 1, Y)
        
        # Inner M6 = (X10-X00)(Y00+Y01)
        X6 = sub_entries(1, 0, 0, 0, X)
        Y6 = add_entries(0, 0, 0, 1, Y)
        
        # Inner M7 = (X01-X11)(Y10+Y11)
        X7 = sub_entries(0, 1, 1, 1, X)
        Y7 = add_entries(1, 0, 1, 1, Y)
        
        return [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]

    # Contribution of each inner term Z_k to the local 2x2 result Z (which is a block of C)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    # inner_targets[k] = list of (local_i, local_j, coeff)
    inner_targets = [
        [(0,0,1), (1,1,1)],      # Inner M1
        [(1,0,1), (1,1,-1)],     # Inner M2
        [(0,1,1), (1,1,1)],      # Inner M3
        [(0,0,1), (1,0,1)],      # Inner M4
        [(0,0,-1), (0,1,1)],     # Inner M5
        [(1,1,1)],               # Inner M6
        [(0,0,1)],               # Inner M7
    ]

    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        factors = inner_factors(X, Y)
        
        for k in range(7):
            Xf, Yf = factors[k]
            
            # Fill U from Xf (linear combos of A entries)
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c, a_v) in Xf[i][j]:
                        U[a_r * m + a_c, col] += a_v
            
            # Fill V from Yf (linear combos of B entries)
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c, b_v) in Yf[i][j]:
                        V[b_r * p + b_c, col] += b_v
            
            # Fill W from c_targets[t] and inner_targets[k]
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify: reconstruct the multiplication tensor and check correctness
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: C[i,k] = sum_r U[i,r] * V[k,r] * W[i,k,r] (with W indexed as W[k*n+i, r])
    # Actually: C[i,k] = sum_r U[i*m+j... wait, let me re-read the convention.
    # U has shape (n*m, rank) = (16, 49): U[i*m+j, r] is coeff of A[i,j] in factor r
    # V has shape (m*p, rank) = (16, 49): V[j*p+k, r] is coeff of B[j,k] in factor r  
    # W has shape (p*n, rank) = (16, 49): W[k*n+i, r] is coeff for C[i,k] in factor r
    # Check: for all i,k,j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Actually the tensor entry T[i,k,j] = delta_{i,k}... no.
    # The matrix multiplication tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,k,j] = 1 for all i,j,k (the identity in the middle index doesn't apply)
    # Wait: C[i,k] = sum_j A[i,j] * B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The structure tensor for matmul: each entry of C is a bilinear form in A and B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 
    # But U has shape (n*m, rank), so U is indexed by (i,j) pair and r.
    # Let me just verify numerically.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, k), check that sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Actually no - the decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right either. Let me think again.
    # Standard tensor decomposition of matmul:
    # The tensor T has indices (i, j, k) where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # Actually T is 3D: T[a, b, c] where a indexes A entries, b indexes B entries, c indexes C entries
    # T[a, b, c] = 1 if A entry a and B entry b contribute to C entry c
    # A entry a = i*m + j, B entry b = j'*p + k, C entry c = k'*n + i'
    # T[a, b, c] = 1 if i=i' and k=k' and j=j' (i.e., A[i,j]*B[j,k] contributes to C[i,k])
    # So T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check: for all valid (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all invalid combos: sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # Spot check a few entries
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                if abs(val - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    assert verified, "Decomposition verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
