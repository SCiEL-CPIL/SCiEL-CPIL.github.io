"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10)."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10

    # We want to find a rank-minimal decomposition of the 2x3x10 matrix multiplication tensor.
    # The tensor T[i, j, k] = 1 if and only if it contributes to C[i,k] = sum_j A[i,j] * B[j,k].
    # Specifically, the multiplication tensor is:
    #   T = sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{9} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where the output index is (k, i) following the trace convention W[k*n+i, r].
    
    # We'll use a randomized approach to find a low-rank decomposition.
    # The idea: if we can find vectors u_r, v_r, w_r such that
    #   T = sum_r u_r ⊗ v_r ⊗ w_r
    # then we have a rank-r decomposition.
    
    # For a 2x3x10 tensor, the trivial rank is 2*3*10 = 60 (one term per entry).
    # We can do better by exploiting structure.
    
    # Let's try a systematic approach:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For each column k, we have a 2x3x1 "slice" which is just a bilinear form.
    # The rank of a 2x3x1 tensor is at most 2 (since the output is 2-dimensional).
    # Actually, for fixed k, the tensor is rank 2: 
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # This can be written as rank 2 if we combine terms appropriately.
    
    # Let's think about this differently. The total tensor has 2*3*10 = 60 non-zero entries.
    # A rank-1 term u ⊗ v ⊗ w can cover multiple entries if u, v, w have multiple non-zero components.
    
    # Strategy: Use the fact that for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k].
    # We can group terms to reduce rank.
    
    # Let's try: for each column k, use rank 2 decomposition:
    #   C[0,k] = (A[0,0]+A[0,1])*B[0,k] + A[0,1]*(B[1,k]-B[0,k]) + A[0,2]*B[2,k]
    #   C[1,k] = (A[1,0]+A[1,1])*B[0,k] + A[1,1]*(B[1,k]-B[0,k]) + A[1,2]*B[2,k]
    
    # Actually, let's just use a direct approach with numpy to find a good decomposition.
    # We'll use the fact that the tensor can be unfolded and we can use SVD-like techniques.
    
    # Unfold the tensor along the output mode:
    # T is a 3D array of shape (2, 3, 10) where T[i,j,k] = 1 if this term exists.
    # Actually, T[i,j,k] = 1 for all i,j,k since every combination contributes.
    # Wait, no. The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    # So T is just an all-ones tensor of shape (2,3,10)? No, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor for (n,m,p) = (2,3,10) is:
    #   T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_{k*n+i}
    # where the third index follows the trace convention.
    
    # So T[i, j, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # This is a 2 x 3 x 20 tensor (since p*n = 20).
    
    # Let's construct this tensor and try to find its rank using randomized methods.
    
    # Construct the tensor T of shape (n, m, p*n) = (2, 3, 20)
    T = np.zeros((n, m, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k*n+i] = 1.0
    
    # Now we want to find rank-r decomposition: T = sum_{r} u_r ⊗ v_r ⊗ w_r
    # where u_r has shape (n,), v_r has shape (m,), w_r has shape (p*n,).
    
    # This is equivalent to: for each (i,j,l), T[i,j,l] = sum_r U[i,r] * V[j,r] * W[l,r]
    
    # Let's use a greedy/randomized approach to find a low-rank decomposition.
    # We'll start with a random initialization and use alternating minimization.
    
    def try_decomposition(rank, max_iter=1000, tol=1e-10, seed=42):
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(n, rank)
        V = rng.randn(m, rank)
        W = rng.randn(p*n, rank)
        
        for it in range(max_iter):
            if time.time() - _t0 > 100:
                return None, None, None, it
            
            # Update U: for each i, U[i,:] = argmin_U sum_{j,l} (T[i,j,l] - sum_r U[i,r]*V[j,r]*W[l,r])^2
            # This is a linear system: for fixed i, we want to minimize
            # sum_{j,l} (T[i,j,l] - U[i,:] @ (V[j,:] * W[l,:]))^2
            # Let M_i be a matrix where M_i[j*n+l, r] = V[j,r] * W[l,r]
            # Then U[i,:] = pinv(M_i) @ T[i,:,:].reshape(-1)
            
            for i in range(n):
                M_i = np.zeros((m*p*n, rank))
                for j in range(m):
                    for l in range(p*n):
                        M_i[j*p*n+l, :] = V[j, :] * W[l, :]
                b = T[i, :, :].reshape(-1)
                # Solve least squares
                U[i, :], residuals, rank_s, sv = np.linalg.lstsq(M_i, b, rcond=None)
            
            # Update V similarly
            for j in range(m):
                M_j = np.zeros((n*p*n, rank))
                for i in range(n):
                    for l in range(p*n):
                        M_j[i*p*n+l, :] = U[i, :] * W[l, :]
                b = T[:, j, :].reshape(-1)
                V[j, :], residuals, rank_s, sv = np.linalg.lstsq(M_j, b, rcond=None)
            
            # Update W similarly
            for l in range(p*n):
                M_l = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        M_l[i*m+j, :] = U[i, :] * V[j, :]
                b = T[:, :, l]
                W[l, :], residuals, rank_s, sv = np.linalg.lstsq(M_l, b, rcond=None)
            
            # Check convergence
            if it % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for l in range(p*n):
                            pred = 0.0
                            for r in range(rank):
                                pred += U[i, r] * V[j, r] * W[l, r]
                            err += (T[i, j, l] - pred) ** 2
                if err < tol:
                    return U, V, W, it
        
        return U, V, W, max_iter
    
    # Try different ranks
    best_rank = 60  # trivial upper bound
    best_U = best_V = best_W = None
    
    for rank in range(20, 61):
        if time.time() - _t0 > 100:
            break
        U, V, W, it = try_decomposition(rank, max_iter=200, seed=rank)
        if U is not None:
            # Verify the decomposition
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for l in range(p*n):
                        pred = 0.0
                        for r in range(rank):
                            pred += U[i, r] * V[j, r] * W[l, r]
                        err += (T[i, j, l] - pred) ** 2
            if err < 1e-6:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    if best_U is None:
        # Fallback: use the trivial rank-60 decomposition
        best_rank = 60
        best_U = np.zeros((n, best_rank))
        best_V = np.zeros((m, best_rank))
        best_W = np.zeros((p*n, best_rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    l = k*n+i
                    best_U[i, r] = 1.0
                    best_V[j, r] = 1.0
                    best_W[l, r] = 1.0
                    r += 1
        best_rank = r
    
    # Convert to lists of lists with integer/half-integer entries if possible
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    # Round to nearest half-integer if close
    def round_half(x):
        rx = round(x * 2) / 2
        return rx if abs(x - rx) < 1e-6 else x
    
    U_list = [[round_half(x) for x in row] for row in U_list]
    V_list = [[round_half(x) for x in row] for row in V_list]
    W_list = [[round_half(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
