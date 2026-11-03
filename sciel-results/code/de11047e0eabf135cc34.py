"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_shape = (n * m,)  # 16
    V_shape = (m * p,)  # 36
    W_shape = (p * n,)  # 36
    
    # Build the target tensor T as a sparse representation or dense array
    # T[a, b, c_idx] = 1 if (a, b, c_idx) is a valid triple
    # a = i*m + j, b = j*p + k, c_idx = k*n + i
    # n=4, m=4, p=9
    # a in [0, 15], b in [0, 35], c_idx in [0, 35]
    
    T = np.zeros((n*m, m*p, p*n))
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
                triples.append((a, b, c_idx))
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for col, (a, b, c_idx) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c_idx, col] = 1
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        # Check all non-zero entries
        for (a, b, c_idx) in triples:
            val = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of zero entries
        # To be rigorous, we should check all, but it's 20k entries.
        # 20k * 130 mults is ~2.6M ops, which is fast.
        # Let's check all to be safe.
        # Create a mask of zero entries
        # Actually, iterating over all indices is slow in Python.
        # We can vectorize the check.
        # T_check = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 16x36x36 array.
        # Then compare with T.
        T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(T_reconstructed, T, atol=tol):
            return False
        return True

    def als_step(U, V, W, T, n_iters=50):
        """Perform ALS optimization."""
        R = U.shape[1]
        # Precompute products for efficiency?
        # For U update:
        # Z_U is (m*p) x R, Z_U[b, c, r] = V[b, r] * W[c, r]
        # This is not a simple matrix product.
        # We can reshape V and W.
        # V is (m*p, R), W is (p*n, R).
        # We need Z_U[b, c, r] for b in 0..mp-1, c in 0..pn-1.
        # Note that b and c are independent indices in the tensor, but in the matrix multiplication,
        # b and c are linked by the structure of the problem?
        # No, in the tensor T, the indices are a, b, c.
        # The constraint is sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        # So for fixed a, we solve for U[a,:].
        # The "features" are vectors z_{bc} where z_{bc}[r] = V[b,r] * W[c,r].
        # We have mp * pn such vectors.
        # We can construct a matrix Z of shape (mp*pn, R).
        # Z[idx, r] = V[b, r] * W[c, r] where idx = b * pn + c.
        
        # Construct Z_U
        # V is (mp, R), W is (pn, R)
        # We need outer product of V and W along the R dimension?
        # No, element-wise product along R, then flatten b,c.
        # V[:, :, None] * W[None, :, :] gives (mp, pn, R).
        # Reshape to (mp*pn, R).
        Z_U = (V[:, :, None] * W[None, :, :]).reshape(-1, R)
        # T_U is T reshaped to (n*m, mp*pn). T[a, b, c] -> T_U[a, b*pn+c]
        T_U = T.reshape(n*m, -1)
        
        # Construct Z_V
        # Features for V[b,r] are U[a,r] * W[c,r].
        # Z_V is (n*m * p*n, R).
        # Z_V[idx, r] = U[a, r] * W[c, r] where idx = a * pn + c.
        Z_V = (U[:, :, None] * W[None, :, :]).reshape(-1, R)
        T_V = T.reshape(m*p, -1).T.reshape(-1, m*p) # Wait, T is (n*m, m*p, p*n).
        # We need T_V[b, a*pn+c] = T[a, b, c].
        # T_V = T.transpose(1, 0, 2).reshape(m*p, -1)
        T_V = T.transpose(1, 0, 2).reshape(m*p, -1)
        
        # Construct Z_W
        # Features for W[c,r] are U[a,r] * V[b,r].
        # Z_W is (n*m * m*p, R).
        # Z_W[idx, r] = U[a, r] * V[b, r] where idx = a * mp + b.
        Z_W = (U[:, :, None] * V[None, :, :]).reshape(-1, R)
        T_W = T.transpose(2, 0, 1).reshape(p*n, -1)
        
        for _ in range(n_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # Solve Z_U.T @ U.T = T_U.T for each row?
            # U.T is (R, n*m).
            # Z_U is (N, R). Z_U.T is (R, N).
            # System: (Z_U.T @ Z_U) @ U.T = Z_U.T @ T_U.T
            # A = Z_U.T @ Z_U (R x R)
            # B = Z_U.T @ T_U.T (R x n*m)
            A_U = Z_U.T @ Z_U
            B_U = Z_U.T @ T_U.T
            try:
                U.T = np.linalg.solve(A_U, B_U)
            except np.linalg.LinAlgError:
                U.T = np.linalg.lstsq(A_U, B_U, rcond=None)[0]
                
            # Update V
            Z_V = (U[:, :, None] * W[None, :, :]).reshape(-1, R)
            A_V = Z_V.T @ Z_V
            B_V = Z_V.T @ T_V.T
            try:
                V.T = np.linalg.solve(A_V, B_V)
            except np.linalg.LinAlgError:
                V.T = np.linalg.lstsq(A_V, B_V, rcond=None)[0]
                
            # Update W
            Z_W = (U[:, :, None] * V[None, :, :]).reshape(-1, R)
            A_W = Z_W.T @ Z_W
            B_W = Z_W.T @ T_W.T
            try:
                W.T = np.linalg.solve(A_W, B_W)
            except np.linalg.LinAlgError:
                W.T = np.linalg.lstsq(A_W, B_W, rcond=None)[0]
                
        return U, V, W

    # Search for lower ranks
    # Lower bound is max(16, 36, 36) = 36.
    # Known optimal is ~130.
    # We try from 130 up to 143.
    for target_rank in range(130, rank_school):
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(target_rank * 100 + attempt)
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            U, V, W = als_step(U, V, W, T, n_iters=50)
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = target_rank
                break
        
        # If we found a rank, we can stop if it's close to lower bound
        # Or continue to see if we can go lower?
        # The loop goes 130 to 143. If 130 works, we break.
        if best_rank == target_rank:
            break

    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
