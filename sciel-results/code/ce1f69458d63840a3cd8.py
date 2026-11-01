import time
import numpy as np
import random

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def verify(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Tensor T = sum_k U[:,k] V[:,k] W[:,k]
        # Check if T matches matrix multiplication tensor.
        # The tensor for M_{2,2,8} has entries T[i,j,k] = 1 if j==k? No.
        # A (2x2) * B (2x8) = C (2x8).
        # C[i, k] = sum_j A[i, j] B[j, k].
        # Tensor indices: i (row of A), j (col of A / row of B), k (col of B).
        # T[i, j, k] = 1 if i is row of A, j is col of A, k is col of B.
        # In the decomposition:
        # C[i, k] = sum_l U[i, l] V[?, l] W[?, l] ...
        # Standard tensor decomposition for M_{n,m,p}:
        # T = sum_r U_r \otimes V_r \otimes W_r
        # U_r is vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
        # The tensor entry T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n).
        # Mapping:
        # a = i * m + j  (i in A row, j in A col)
        # b = j * p + k  (j in B row, k in B col)
        # c = k * n + i  (k in C col, i in C row)
        # Note the trace convention: W[k*n+i, r].
        
        rank = U.shape[1]
        # Compute the decomposed tensor
        # T[ a, b, c ] = sum_r U[a, r] * V[b, r] * W[c, r]
        
        # We only need to check non-zero entries of the target tensor.
        # Target tensor T_target:
        # T_target[a, b, c] = 1 if:
        #   a = i*m + j
        #   b = j*p + k
        #   c = k*n + i
        #   AND the index j in a matches the index j in b.
        
        # Efficient check:
        # For each r, we add U[:,r] V[:,r] W[:,r] to T.
        # This is O(rank * (n*m) * (m*p) * (p*n)) which is too slow if done naively for large rank.
        # But here dimensions are small.
        # n*m = 4, m*p = 16, p*n = 16.
        # Total entries = 4 * 16 * 16 = 1024.
        # Rank <= 28.
        # 28 * 1024 = 28k operations. Very fast.
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T += np.einsum('i,j,k->ijk', u, v, w)
            
        # Check against target
        # Iterate through all possible (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Target value is 1
                    if abs(T[a, b, c] - 1.0) > 1e-6:
                        return False
        # Check that other entries are 0?
        # The tensor M_{n,m,p} is 0 for invalid index couplings.
        # We should verify zeros too.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This covers all (a,b,c) that are coupled.
                    # We need to check all 1024 entries.
                    pass
        
        # Full check
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Determine if this entry should be 1 or 0
                    # a = i*m + j => i = a//m, j = a%m
                    # b = j'*p + k => j' = b//p, k = b%p
                    # c = k'*n + i' => k' = c//n, i' = c%n
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    
                    if i == ip and j == jp and k == kp:
                        if abs(T[a, b, c] - 1.0) > 1e-6:
                            return False
                    else:
                        if abs(T[a, b, c] - 0.0) > 1e-6:
                            return False
        return True

    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        strassen_products = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        return U, V, W

    def try_reduce_rank(U, V, W, target_rank, max_iters=5000):
        """Try to reduce rank by finding linear dependencies."""
        rank = U.shape[1]
        if rank <= target_rank:
            return U, V, W, rank
        
        # Compute the tensor
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Try to find a column that is approximately a linear combination of others
        for iter in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random column to try to eliminate
            col_to_remove = random.randint(0, rank - 1)
            
            # Try to express this column as a linear combination of others
            other_cols = [c for c in range(rank) if c != col_to_remove]
            
            # Build the system: find alpha such that
            # sum_{c in other_cols} alpha[c] * (U[:,c] V[:,c] W[:,c]) = U[:,col_to_remove] V[:,col_to_remove] W[:,col_to_remove]
            # This is a linear system in alpha, but the equations are nonlinear in U,V,W.
            # Instead, let's try a simpler approach: perturb and re-solve.
            
            # Alternative: try to set one entry to zero and adjust others
            # Pick a random nonzero entry in U, V, or W for the column to remove
            matrices = [U, V, W]
            shapes = [(n*m,), (m*p,), (p*n,)]
            
            mat_idx = random.randint(0, 2)
            mat = matrices[mat_idx]
            shape = shapes[mat_idx]
            
            # Find nonzero entries in this column
            col_vals = mat[:, col_to_remove]
            nz = np.nonzero(col_vals)[0]
            if len(nz) == 0:
                continue
            
            # Try to zero out one entry
            idx = nz[random.randint(0, len(nz) - 1)]
            old_val = mat[idx, col_to_remove]
            mat[idx, col_to_remove] = 0.0
            
            # Now we need to compensate by adjusting other columns
            # Compute the error introduced
            error = np.zeros((n*m, m*p, p*n))
            u = U[:, col_to_remove].copy()
            v = V[:, col_to_remove].copy()
            w = W[:, col_to_remove].copy()
            # The old contribution
            old_contrib = np.einsum('i,j,k->ijk', u, v, w)
            # The new contribution (with zeroed entry)
            if mat_idx == 0:
                u[idx] = 0
            elif mat_idx == 1:
                v[idx] = 0
            else:
                w[idx] = 0
            new_contrib = np.einsum('i,j,k->ijk', u, v, w)
            error = old_contrib - new_contrib  # This is what we need to add back
            
            # Distribute the error among other columns
            # Pick a random other column to absorb the error
            other_col = other_cols[random.randint(0, len(other_cols) - 1)]
            
            # We need to add error to the sum, which means adjusting U,V,W for other_col
            # This is hard to do exactly. Instead, let's just check if the error is small.
            if np.abs(error).max() < 1e-10:
                # We successfully removed the entry without error
                # Now check if the column can be removed entirely
                if np.allclose(U[:, col_to_remove], 0) and np.allclose(V[:, col_to_remove], 0) and np.allclose(W[:, col_to_remove], 0):
                    # Remove this column
                    mask = [c for c in range(rank) if c != col_to_remove]
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    rank -= 1
                    if rank <= target_rank:
                        return U, V, W, rank
                    other_cols = [c for c in range(rank) if c != col_to_remove]
                    continue
            
            # Revert the change
            mat[idx, col_to_remove] = old_val
        
        return U, V, W, rank

    # First, get the rank-28 solution
    U, V, W = get_strassen_28()
    best_rank = 28
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    for target in [27, 26, 25, 24]:
        if time.time() > deadline - 20:
            break
        
        U_try, V_try, W_try, rank_try = try_reduce_rank(U.copy(), V.copy(), W.copy(), target, max_iters=2000)
        
        if rank_try < best_rank and verify(U_try, V_try, W_try):
            best_rank = rank_try
            best_U, best_V, best_W = U_try, V_try, W_try
            print(f"Found rank {best_rank} solution")
    
    # If we found a better solution, use it
    if best_rank < 28:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = get_strassen_28()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
