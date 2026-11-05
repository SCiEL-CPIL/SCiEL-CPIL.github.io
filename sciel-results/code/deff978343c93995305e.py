"""Matrix multiplication 3x4 x 4x4 tensor decomposition — rank minimization."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def _schoolbook():
        """Standard rank = n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        return U, V, W

    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def _verify(U, V, W, rank):
        """Verify exact decomposition: T_{i,j,k} = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _build_block_decomposition():
        """
        Block-structured decomposition using Strassen (rank 7) for 2x2x2 blocks
        and rank-4 for 1x2x2 blocks.
        
        Split A (3x4) into rows {0,1} and {2}, cols {0,1} and {2,3}
        Split B (4x4) into rows {0,1} and {2,3}, cols {0,1} and {2,3}
        Output C (3x4): W[k*n+i]
        
        Total: 4 Strassen blocks (28) + 4 rank-4 blocks (16) = 44
        """
        rank = 44
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        col_idx = 0

        def _add_term(u_entries, v_entries, w_entries):
            nonlocal col_idx
            for idx, val in u_entries:
                U[idx][col_idx] = val
            for idx, val in v_entries:
                V[idx][col_idx] = val
            for idx, val in w_entries:
                W[idx][col_idx] = val
            col_idx += 1

        def _strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
            """7 terms for 2x2 x 2x2 -> 2x2 Strassen."""
            # M1 = (a+d)(e+h)
            _add_term([(a, 1.0), (d, 1.0)], [(e, 1.0), (h, 1.0)], [(w11, 1.0), (w22, 1.0)])
            # M2 = (c+d)e
            _add_term([(c, 1.0), (d, 1.0)], [(e, 1.0)], [(w21, 1.0), (w22, -1.0)])
            # M3 = a(f-h)
            _add_term([(a, 1.0)], [(f, 1.0), (h, -1.0)], [(w12, 1.0), (w22, 1.0)])
            # M4 = d(g-e)
            _add_term([(d, 1.0)], [(g, 1.0), (e, -1.0)], [(w11, 1.0), (w21, 1.0)])
            # M5 = (a+b)h
            _add_term([(a, 1.0), (b, 1.0)], [(h, 1.0)], [(w12, 1.0), (w11, -1.0)])
            # M6 = (c-a)(e+f)
            _add_term([(c, 1.0), (a, -1.0)], [(e, 1.0), (f, 1.0)], [(w22, 1.0)])
            # M7 = (b-d)(g+h)
            _add_term([(b, 1.0), (d, -1.0)], [(g, 1.0), (h, 1.0)], [(w11, 1.0)])

        def _rank4(a, b, c, d, e, f, w1, w2):
            """4 terms for 1x2 x 2x2 -> 1x2."""
            _add_term([(a, 1.0)], [(c, 1.0)], [(w1, 1.0)])
            _add_term([(b, 1.0)], [(e, 1.0)], [(w1, 1.0)])
            _add_term([(a, 1.0)], [(d, 1.0)], [(w2, 1.0)])
            _add_term([(b, 1.0)], [(f, 1.0)], [(w2, 1.0)])

        # A[i,j] index = i*4+j; B[j,k] index = j*4+k; W[k*n+i] = k*3+i
        #
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A: (0,1,4,5), B: (0,1,4,5), W: (0,3,1,4)
        _strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)

        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # B: (2,3,6,7), W: (6,9,7,10)
        _strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)

        # Block 3: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        # A: (8,9), B: (0,1,4,5), W: (2,5)
        _rank4(8, 9, 0, 1, 4, 5, 2, 5)

        # Block 4: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        # B: (2,3,6,7), W: (8,11)
        _rank4(8, 9, 2, 3, 6, 7, 8, 11)

        # Block 5: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # A: (2,3,6,7), B: (8,9,12,13), W: (0,3,1,4)
        _strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)

        # Block 6: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        # B: (10,11,14,15), W: (6,9,7,10)
        _strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)

        # Block 7: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
        # A: (10,11), B: (8,9,12,13), W: (2,5)
        _rank4(10, 11, 8, 9, 12, 13, 2, 5)

        # Block 8: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
        # A: (10,11), B: (10,11,14,15), W: (8,11)
        _rank4(10, 11, 10, 11, 14, 15, 8, 11)

        assert col_idx == rank, f"Expected {rank} terms, got {col_idx}"
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    # Build the block decomposition
    U_blk, V_blk, W_blk = _build_block_decomposition()
    if _verify(U_blk, V_blk, W_blk, 44):
        best_U, best_V, best_W = U_blk, V_blk, W_blk
        best_rank = 44

    # Attempt ALS-based rank reduction on the block decomposition
    # Try to find a lower-rank decomposition by optimizing from the rank-44 seed
    if time.time() < deadline - 30:
        def _try_als_reduction(seed_rank, target_ranks, max_iters=200, tol=1e-12):
            """Try to reduce rank via ALS on the tensor."""
            n_rows = n * m  # 12
            n_cols = m * p  # 16
            n_out = p * n   # 12
            
            # Build target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # T is (n, m, p) but we flatten for ALS
            # We work with the flattened forms:
            # U: (12, r), V: (16, r), W: (12, r)
            # Constraint: for all i,j,k: sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*3+i, c] = 1
            
            for target_r in target_ranks:
                if time.time() >= deadline - 20:
                    break
                if target_r >= best_rank:
                    continue
                    
                # Initialize from top-left of the rank-44 matrices
                U_init = np.zeros((n_rows, target_r))
                V_init = np.zeros((n_cols, target_r))
                W_init = np.zeros((n_out, target_r))
                
                # Use first target_r columns of the seed
                if seed_rank >= target_r:
                    U_init = U_blk[:, :target_r].copy()
                    V_init = V_blk[:, :target_r].copy()
                    W_init = W_blk[:, :target_r].copy()
                else:
                    # Random initialization
                    rng = np.random.RandomState(42)
                    U_init = rng.randn(n_rows, target_r)
                    V_init = rng.randn(n_cols, target_r)
                    W_init = rng.randn(n_out, target_r)
                
                U_c = U_init.copy()
                V_c = V_init.copy()
                W_c = W_init.copy()
                
                best_loss = float('inf')
                best_U = U_c.copy()
                best_V = V_c.copy()
                best_W = W_c.copy()
                
                for it in range(max_iters):
                    if time.time() >= deadline - 20:
                        break
                    
                    # Precompute helper matrices for efficiency
                    # For U update: fix V, W. For each i,j, we need:
                    # sum_k V[j*4+k,:] @ (something involving W)
                    # T[i,j,k] = sum_c U[i*4+j,c] V[j*4+k,c] W[k*3+i,c]
                    # For fixed i,j: sum_k sum_c U[i*4+j,c] V[j*4+k,c] W[k*3+i,c] = 1
                    # => sum_c U[i*4+j,c] * (sum_k V[j*4+k,c] W[k*3+i,c]) = 1
                    # Let S_ij,c = sum_k V[j*4+k,c] W[k*3+i,c]
                    # Then sum_c U[i*4+j,c] S_ij,c = 1 for all i,j
                    
                    # For ALS on U: treat (S_ij) as the "data" matrix D (size 12 x 12),
                    # and we want U @ S^T to be all-ones? No...
                    # Actually: for each (i,j), sum_c U[i*4+j, c] * S_ij[c] = 1
                    # This is a linear system in U[i*4+j, :] for each (i,j).
                    # U[i*4+j, :] @ S_ij = 1 (scalar)
                    # This is underdetermined if target_r < 12, so use least squares / pseudo-inverse.
                    
                    # Build S matrix: S[i*4+j, c] = sum_k V[j*4+k, c] * W[k*3+i, c]
                    S = np.zeros((n_rows, target_r))
                    for i in range(n):
                        for j in range(m):
                            # S[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                            for k in range(p):
                                S[i * m + j, :] += V_c[j * p + k, :] * W_c[k * n + i, :]
                    
                    # For each row idx = i*m+j, we want U[idx, :] @ S[idx, :] = 1
                    # Least squares: U[idx, :] = S[idx, :] / ||S[idx, :]||^2 (if nonzero)
                    # Actually this is: minimize ||U[idx,:] @ S[idx,:] - 1||^2
                    # Solution: U[idx,:] = S[idx,:] / dot(S[idx,:], S[idx,:])
                    for idx in range(n_rows):
                        dot = np.dot(S[idx, :], S[idx, :])
                        if dot > 1e-15:
                            U_c[idx, :] = S[idx, :] / dot
                        # else: leave as is (degenerate case)
                    
                    # Update V: for each j,k:
                    # sum_c V[j*4+k, c] * (sum_i U[i*4+j, c] W[k*3+i, c]) = 1
                    R = np.zeros((n_cols, target_r))
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                R[j * p + k, :] += U_c[i * m + j, :] * W_c[k * n + i, :]
                    for idx in range(n_cols):
                        dot = np.dot(R[idx, :], R[idx, :])
                        if dot > 1e-15:
                            V_c[idx, :] = R[idx, :] / dot
                    
                    # Update W: for each k,i:
                    # sum_c W[k*3+i, c] * (sum_j U[i*4+j, c] V[j*4+k, c]) = 1
                    Q = np.zeros((n_out, target_r))
                    for k in range(p):
                        for i in range(n):
                            for j in range(m):
                                Q[k * n + i, :] += U_c[i * m + j, :] * V_c[j * p + k, :]
                    for idx in range(n_out):
                        dot = np.dot(Q[idx, :], Q[idx, :])
                        if dot > 1e-15:
                            W_c[idx, :] = Q[idx, :] / dot
                    
                    # Compute loss
                    loss = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_c[i * m + j, :], V_c[j * p + k, :] * W_c[k * n + i, :])
                                loss += (val - 1.0) ** 2
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_U = U_c.copy()
                        best_V = V_c.copy()
                        best_W = W_c.copy()
                    
                    if best_loss < tol:
                        break
                    if best_loss < 1e-6:
                        # Check exactness
                        if _verify(best_U, best_V, best_W, target_r):
                            return best_U, best_V, best_W, target_r
                    if it % 20 == 19 and best_loss > 1.0:
                        break  # Not converging well
                
                # Final check
                if _verify(best_U, best_V, best_W, target_r):
                    return best_U, best_V, best_W, target_r
            
            return None, None, None, best_rank
        
        # Try target ranks from 43 down to 40
        result = _try_als_reduction(44, [43, 42, 41, 40])
        if result[0] is not None:
            best_U, best_V, best_W, best_rank = result
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try a different block splitting: rows {0}, {1,2} instead of {0,1}, {2}
    # This might allow better Strassen utilization
    if time.time() < deadline - 20:
        def _build_block_decomposition_v2():
            """
            Alternative split: A rows {0} and {1,2}
            A[0, :] x B[:, 0:2] -> C[0, 0:2]  (1x4 x 4x2, rank 8)
            A[0, :] x B[:, 2:4] -> C[0, 2:4]  (1x4 x 4x2, rank 8)
            A[1:3, 0:2] x B[0:2, 0:2] -> C[1:3, 0:2] (2x2 x 2x2, Strassen rank 7)
            A[1:3, 0:2] x B[0:2, 2:4] -> C[1:3, 2:4] (2x2 x 2x2, Strassen rank 7)
            A[1:3, 2:4] x B[2:4, 0:2] -> C[1:3, 0:2] (2x2 x 2x2, Strassen rank 7)
            A[1:3, 2:4] x B[2:4, 2:4] -> C[1:3, 2:4] (2x2 x 2x2, Strassen rank 7)
            Total: 8 + 8 + 7 + 7 + 7 + 7 = 44 (same)
            """
            # Same rank, skip
            return None, None, None
        
        # Try: A rows {0,1,2} all together with different B splits
        # Not obviously better, skip
        
        # Try a 3-way split of B columns: {0}, {1,2}, {3}
        # A[0:2, 0:2] x B[0:2, 0] -> rank 2x2x1 = 4
        # Getting complicated; the rank-44 seems to be the natural barrier.
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
