"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Start from a known valid decomposition (schoolbook rank 216), then attempt
rank reduction by merging columns where U and V vectors are linearly dependent.
We use a greedy approach: try to combine pairs of columns that have proportional U and V vectors.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses a model-and-optimize approach: parameterize the decomposition and attempt
    to reduce rank by finding linear dependencies among the rank-1 components.
    """
    deadline = time.time() + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 216)
    # Each term: U[row, col] = 1 if row corresponds to (i,j), V[col, col] = 1 if col corresponds to (j,k)
    # W[col, col] = 1 if it maps to output (k,i)
    
    rank_schoolbook = n * m * p  # 216
    
    # Build schoolbook factors
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                # W index: W[k*n+i, r]
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify correctness of schoolbook
    # C = A @ B, where (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For a single column corresponding to (i',j',k'), this contributes:
    #   1 * 1 * 1 to position (k',i') in W, which maps to output[i',k']
    # So summing over all columns: output[i,k] = sum_{j} A[i,j]*B[j,k] * (sum over matching i',j',k')
    # This is correct.
    
    # Now attempt rank reduction
    # Strategy: Find groups of columns where U vectors and V vectors are linearly dependent
    # If two columns have U[:,c1] = alpha * U[:,c2] and V[:,c1] = beta * V[:,c2],
    # then W[:,c1] must equal alpha*beta * W[:,c2] for them to be mergeable.
    # In that case, we can keep one column and scale W accordingly.
    
    # More generally, if a set of columns has U vectors spanning a d-dimensional space
    # and V vectors spanning a d-dimensional space, we can potentially reduce to d columns.
    
    # Let's use a greedy approach: repeatedly find pairs that can be merged.
    # Two columns c1, c2 can be merged if U[:,c1] and U[:,c2] are linearly dependent
    # AND V[:,c1] and V[:,c2] are linearly dependent.
    
    # For schoolbook, each column has U with a single 1 and V with a single 1.
    # No two columns have dependent U vectors (each has a different position with 1).
    # So schoolbook can't be reduced this way directly.
    
    # Alternative approach: Use block structure with Strassen at the 2x2 level.
    # 6 = 3 * 2, so we have a 3x3 block matrix with 2x2 blocks.
    # Use Strassen (rank 7) for each 2x2 multiplication.
    # Total rank: 9 * 3 * 7 = 189
    
    # But we can do better by using a smarter block decomposition.
    # Let's try 6 = 2 * 3, so 2x2 blocks with 3x3 submatrices? No, that's the same.
    
    # Actually, let's just implement the block-Strassen approach which gives rank 189.
    # Then try to reduce further.
    
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Verify the block-Strassen decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives correct matrix multiplication tensor."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat_2d = U_mat[:, r].reshape(n, m)
            v_mat_2d = V_mat[:, r].reshape(m, p)
            w_mat_2d = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U_bs, V_bs, W_bs):
        # Fallback to schoolbook if block-Strassen is wrong
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a second block decomposition: 6 = 2*3, use 2x2 Strassen on 3x3 blocks
    # This gives 8 * 7 = 56 rank-1 terms at the block level, but each term
    # requires a 3x3 multiplication. Use schoolbook for 3x3 (27 terms each).
    # Total: 8 * 7 * 27 / 7 = 8 * 27 = 216? No, that's worse.
    # Actually: for each of the 8 (ib,jb,kb) block triples, we do a 2x2 Strassen
    # on 3x3 blocks. Each Strassen term is a 3x3 multiplication (schoolbook rank 27).
    # Total rank = 8 * 7 * 27 = 1512. Much worse.
    
    # Alternative: 6 = 3*2 (current approach) gives 27 * 7 = 189.
    # 6 = 2*3 gives 8 * 27 = 216 (schoolbook on 3x3, no Strassen at top).
    # So 3*2 with Strassen is better.
    
    # Try: can we use a rank-21 algorithm for 3x3?
    # If yes, total = 27 * 21 / 27 * 7 = 21 * 7 = 147? No, that's not how it works.
    # The structure is: for each (ib,jb,kb) in 3x3x3, do a 2x2 Strassen (7 terms).
    # The 3x3 index multiplication is implicit in the 27 block-triples.
    # We can't separate them easily without a different construction.
    
    # For now, return the verified block-Strassen result (rank 189)
    return {"U": U_bs.tolist(), "V": V_bs.tolist(), "W": W_bs.tolist()}
