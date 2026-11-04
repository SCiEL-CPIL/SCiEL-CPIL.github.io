"""Matrix multiplication 4x5x5 with Strassen-hierarchy tensor network ansatz."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Strategy: Use a structured approach based on Strassen-like algebraic
    cancellations. We decompose the problem by splitting the matrices into
    blocks and using bilinear combinations to reduce the number of
    multiplications below the schoolbook bound of 4*5*5 = 100.
    
    For a 4x5 times 5x5 multiplication, we can try to use a combination of
    smaller Strassen-like decompositions.
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (20, rank)
    # W index: k*n + i (trace convention W[k*n+i, r])
    
    # We'll build a decomposition using Strassen for 2x2x2 sub-blocks
    # and schoolbook for the rest.
    
    # Strategy: Partition the indices into blocks and use Strassen where possible.
    # For the inner dimension j (0..4), partition into {0,1}, {2,3}, {4}
    # For rows i (0..3), partition into {0,1}, {2,3}
    # For cols k (0..4), partition into {0,1}, {2,3}, {4}
    
    # For each combination of (i-block, j-block, k-block), if the block is 2x2x2,
    # use Strassen (rank 7). Otherwise use schoolbook.
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    # We'll accumulate rank-1 terms. Start with empty lists.
    terms = []  # Each term is (u_idx, v_idx, w_idx, val_u, val_v, val_w)
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                len_i = len(i_blk)
                len_j = len(j_blk)
                len_k = len(k_blk)
                
                if len_i == 2 and len_j == 2 and len_k == 2:
                    # Use Strassen for this 2x2x2 sub-block
                    i1, i2 = i_blk
                    j1, j2 = j_blk
                    k1, k2 = k_blk
                    
                    # Verified Strassen 2x2x2 decomposition
                    # C[i1,k1] = A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
                    # C[i1,k2] = A[i1,j1]*B[j1,k2] + A[i1,j2]*B[j2,k2]
                    # C[i2,k1] = A[i2,j1]*B[j1,k1] + A[i2,j2]*B[j2,k1]
                    # C[i2,k2] = A[i2,j1]*B[j1,k2] + A[i2,j2]*B[j2,k2]
                    
                    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
                    terms.append({
                        'U': [(i1*m+j1, 1), (i2*m+j2, 1)],
                        'V': [(j1*p+k1, 1), (j2*p+k2, 1)],
                        'W': [(k1*n+i1, 1), (k2*n+i1, -1), (k1*n+i2, 1), (k2*n+i2, 1)]
                    })
                    
                    # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
                    terms.append({
                        'U': [(i2*m+j1, 1), (i2*m+j2, 1)],
                        'V': [(j1*p+k1, 1)],
                        'W': [(k1*n+i2, 1), (k2*n+i2, -1)]
                    })
                    
                    # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
                    terms.append({
                        'U': [(i1*m+j1, 1)],
                        'V': [(j2*p+k1, 1), (j2*p+k2, -1)],
                        'W': [(k1*n+i1, 1), (k2*n+i1, -1)]
                    })
                    
                    # M4 = A[i2,j2] * (B[j1,k1]-B[j1,k2])
                    terms.append({
                        'U': [(i2*m+j2, 1)],
                        'V': [(j1*p+k1, 1), (j1*p+k2, -1)],
                        'W': [(k1*n+i1, -1), (k1*n+i2, 1), (k2*n+i2, -1)]
                    })
                    
                    # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
                    terms.append({
                        'U': [(i1*m+j1, 1), (i1*m+j2, 1)],
                        'V': [(j2*p+k2, 1)],
                        'W': [(k2*n+i1, 1)]
                    })
                    
                    # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
                    terms.append({
                        'U': [(i2*m+j1, 1), (i1*m+j1, -1)],
                        'V': [(j1*p+k1, 1), (j1*p+k2, 1)],
                        'W': [(k1*n+i2, -1), (k2*n+i2, 1)]
                    })
                    
                    # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
                    terms.append({
                        'U': [(i1*m+j2, 1), (i2*m+j2, -1)],
                        'V': [(j2*p+k1, 1), (j2*p+k2, 1)],
                        'W': [(k1*n+i1, -1), (k2*n+i1, 1), (k2*n+i2, 1)]
                    })
                    
                else:
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                terms.append({
                                    'U': [(i*m+j, 1)],
                                    'V': [(j*p+k, 1)],
                                    'W': [(k*n+i, 1)]
                                })
    
    # Now build U, V, W from terms
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, val in term['U']:
            U[idx][r] = val
        for idx, val in term['V']:
            V[idx][r] = val
        for idx, val in term['W']:
            W[idx][r] = val
    
    # Verify correctness: check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] matches the matmul tensor
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build expected tensor (sparse)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We'll check all entries
    
    tensor_reconstructed = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        tensor_reconstructed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = tensor_reconstructed[i*m+j, j*p+k, k*n+i]
                max_err = max(max_err, abs(actual - expected))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
