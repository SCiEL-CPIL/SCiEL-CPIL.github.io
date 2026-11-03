import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if j matches between A and B, etc.
    # T is a 3x3x3 tensor.
    # Flatten T to a matrix of size (n*m) x (m*p * p*n)? No.
    # Let's use the standard flattening for ALS.
    # T_{i, j, k} where i is index in A (0..8), jk is index in B (0..8) and C (0..8)?
    # No, the decomposition is U (n*m x R), V (m*p x R), W (p*n x R).
    # The tensor entry T[i, j, k] corresponds to U[i, r], V[j, r], W[k, r]?
    # Let's check the indices again.
    # U index: i*m + j. (i: 0..n-1, j: 0..m-1)
    # V index: j*p + k. (j: 0..m-1, k: 0..p-1)
    # W index: k*n + i. (k: 0..p-1, i: 0..n-1)
    
    # The tensor T has non-zero entries at (u_idx, v_idx, w_idx)
    # where u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i.
    # And the value is 1.
    
    T_flat = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_flat[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        # Check if U, V, W exactly decompose T_flat
        # Reconstruct T_rec
        # T_rec[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # We can do this efficiently using einsum or loops.
        # Given small sizes, a loop is fine, or np.einsum
        R = U.shape[1]
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_rec, T_flat, atol=1e-9)

    def round_and_verify(U, V, W):
        # Round to nearest integer or half-integer?
        # The problem says "integer/half-integer".
        # Let's try rounding to nearest 0.5.
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    def als_step(U, V, W, T_flat):
        # Update U
        # For each u, T_flat[u, :, :] = sum_r U[u, r] * (V[:, r] * W[:, r])
        # Let Z_r = V[:, r] * W[:, r] (element-wise product, flattened to 1D? No, V and W are vectors in the decomposition sense?
        # No, V is (m*p, R), W is (p*n, R).
        # The term is V[v, r] * W[w, r].
        # So for fixed r, the matrix M_r = outer(V[:, r], W[:, r])?
        # No, V[:, r] is a vector of length m*p. W[:, r] is a vector of length p*n.
        # The product V[v, r] * W[w, r] is a scalar for each v, w.
        # So the contribution of term r to the slice T_flat[:, v, w] is U[:, r] * (V[v, r] * W[w, r]).
        # This is not a standard matrix outer product.
        # Let's reshape T_flat to a matrix T_mat of size (n*m) x (m*p * p*n).
        # T_mat[u, v * (p*n) + w] = T_flat[u, v, w]
        # Then T_mat = sum_r U[:, r] * (V[:, r] * W[:, r])^T ?
        # Let Z_r be the vector formed by element-wise product of V[:, r] and W[:, r]?
        # No, V[:, r] and W[:, r] have different lengths (m*p vs p*n).
        # The term is V[v, r] * W[w, r].
        # This is not an outer product of two vectors unless we flatten the pair (v,w).
        # Let's define Z_r as a vector of length (m*p * p*n) where Z_r[v * (p*n) + w] = V[v, r] * W[w, r].
        # Then T_mat = U @ Z^T ?
        # T_mat[u, idx] = sum_r U[u, r] * Z_r[idx].
        # Yes, this works.
        
        # Precompute Z for current V, W
        Z = np.zeros((R, m*p * p*n))
        for r in range(R):
            # V[:, r] is length m*p. W[:, r] is length p*n.
            # We need the vector of products V[v, r] * W[w, r] for all v, w.
            # This is the outer product of V[:, r] and W[:, r] flattened.
            Z[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        
        T_mat = T_flat.reshape(n*m, -1)
        
        # Solve for U: T_mat = U @ Z^T
        # U = T_mat @ Z (Z @ Z^T)^{-1}
        # Use lstsq
        U_new, _, _, _ = np.linalg.lstsq(Z.T, T_mat.T, rcond=None)
        U_new = U_new.T
        
        # Update V
        # T_flat[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # For fixed v, T_flat[:, v, :] = sum_r V[v, r] * (U[:, r] * W[:, r])
        # Let Y_r = U[:, r] * W[:, r] (element-wise, length n*m? No, U is n*m, W is p*n. They don't match).
        # Wait, U[:, r] is length n*m. W[:, r] is length p*n.
        # The product U[u, r] * W[w, r] is not an outer product.
        # It's a scalar for each u, w.
        # So for fixed r, the matrix M_r[u, w] = U[u, r] * W[w, r].
        # This is the outer product of U[:, r] and W[:, r].
        # So T_flat[:, v, :] (size n*m x p*n) = sum_r V[v, r] * outer(U[:, r], W[:, r]).
        # Let Y_r = outer(U[:, r], W[:, r]).flatten() (length n*m * p*n).
        # Then T_flat[:, v, :].flatten() = sum_r V[v, r] * Y_r.
        # So T_slice = V @ Y^T ?
        # T_slice is (p*n) x (n*m)? No.
        # Let's reshape T_flat to (m*p) x (n*m * p*n).
        # T_mat2[v, u * (p*n) + w] = T_flat[u, v, w].
        # Then T_mat2 = V @ Y^T where Y[r, :] = outer(U[:, r], W[:, r]).flatten().
        
        Y = np.zeros((R, n*m * p*n))
        for r in range(R):
            Y[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
        T_mat2 = np.zeros((m*p, n*m * p*n))
        # T_flat is (n*m, m*p, p*n).
        # We want T_mat2[v, idx] where idx corresponds to u, w.
        # T_flat[u, v, w] -> T_mat2[v, u * p*n + w]
        T_mat2 = np.transpose(T_flat, (1, 0, 2)).reshape(m*p, -1)
        
        V_new, _, _, _ = np.linalg.lstsq(Y.T, T_mat2.T, rcond=None)
        V_new = V_new.T
        
        # Update W
        # T_flat[u, v, w] = sum_r W[w, r] * (U[u, r] * V[v, r])
        # For fixed w, T_flat[:, :, w] (n*m x m*p) = sum_r W[w, r] * outer(U[:, r], V[:, r]).
        # Let X_r = outer(U[:, r], V[:, r]).flatten() (length n*m * m*p).
        # T_mat3[w, idx] = T_flat[u, v, w] where idx = u * m*p + v.
        T_mat3 = np.transpose(T_flat, (2, 0, 1)).reshape(p*n, -1)
        
        X = np.zeros((R, n*m * m*p))
        for r in range(R):
            X[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
        W_new, _, _, _ = np.linalg.lstsq(X.T, T_mat3.T, rcond=None)
        W_new = W_new.T
        
        return U_new, V_new, W_new

    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    best_rank = 27
    best_solution = None
    
    # Try ranks from 23 to 26
    for target_rank in [23, 24, 25, 26]:
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for each rank
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            # Use small random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Run ALS
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                U, V, W = als_step(U, V, W, T_flat)
                
                # Check convergence / exactness
                # Calculate error
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                error = np.linalg.norm(T_rec - T_flat)
                
                if error < 1e-7:
                    # Try to round
                    res = round_and_verify(U, V, W)
                    if res:
                        best_rank = target_rank
                        best_solution = res
                        # Return immediately if we found a low rank
                        if target_rank == 23:
                            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
                        break

        if best_solution:
            # If we found a solution for this rank, we can return it
            # But we want the minimal rank, so we continue if we haven't hit 23
            if target_rank == 23:
                break
            # If we found 24, we might still find 23.
            # But time is limited.
            # If we found 24, let's keep it as best_solution.
            # We will continue to 23 if time permits.
            
    if best_solution:
        return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
