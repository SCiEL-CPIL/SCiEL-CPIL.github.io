import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Strategy: Use block decomposition with Strassen-like recursion for 2x2 blocks
    # Partition the 3x4 matrix into blocks and 4x7 matrix into blocks
    # 
    # For 3x4 * 4x7:
    # We can partition rows of A (3x4) into [2, 1] and columns into [2, 2]
    # Partition rows of B (4x7) into [2, 2] and columns into [4, 3]
    # 
    # Actually, let's think about this differently.
    # The standard approach for non-power-of-2 dimensions is to use the best known
    # tensor rank for smaller subproblems and combine them.
    #
    # Key insight: For 2x2x2 matrix multiplication, rank is 7 (Strassen)
    # For 2x1x1, 1x2x1, 1x1x2, rank is 1 (trivial)
    # For 2x2x1, rank is 4 (schoolbook, no known better)
    # For 2x1x2, rank is 4
    # For 1x2x2, rank is 4
    #
    # Let's try a direct approach: use random initialization with gradient descent
    # to find a lower rank decomposition, using the residual as objective.
    
    def make_residual_func(R):
        """Create residual function for rank R decomposition."""
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        def residual():
            # Compute the tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
            # The matmul tensor is: T[i,j,k] = 1 if i=m*row+j_col, j=m*row2+k_col, 
            # k=p*row2+row, and j_col == row2
            # More precisely: A is n x m, B is m x p, C = A*B is n x p
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: T[i,j,k] = delta(j, j) where the contraction index matches
            # Actually: T[i,j,k] = 1 if the middle indices match, i.e., 
            # if we write i = i1*m + i2 (row, col of A), j = j1*p + j2 (row, col of B),
            # k = k1*n + k2 (row, col of C), then T[i,j,k] = 1 if i2 == j1 and k1 == i1 and k2 == j2
            # Wait, let me re-derive.
            # A has shape (n, m), so index (i, j) with i in [0,n), j in [0,m)
            # B has shape (m, p), so index (j, k) with j in [0,m), k in [0,p)
            # C = A*B has shape (n, p), so index (i, k)
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T has indices (i, j, k) where T[i,j,k] represents the coefficient
            # for A[i,j] * B[j,k] in C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
            # But the decomposition is: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # where i ranges over n*m, j over m*p, k over p*n.
            # The "true" tensor T[i,j,k] = 1 if the indices are consistent with matmul.
            # Actually, for matrix multiplication, the tensor is:
            # T[i,j,k] = 1 if i = row_A, j = col_A * p + row_B... no.
            
            # Let me use the standard formulation:
            # A is n x m: elements A[a, b] for a in [0,n), b in [0,m)
            # B is m x p: elements B[c, d] for c in [0,m), d in [0,p)
            # C is n x p: elements C[a, d] = sum_b A[a,b] * B[b,d]
            # 
            # Flatten: 
            # A index: a*m + b
            # B index: c*p + d  
            # C index: d*n + a
            # 
            # The tensor T[A_idx, B_idx, C_idx] = 1 if b == c (the contraction index matches)
            # So T[a*m+b, c*p+d, d*n+a] = 1 if b == c
            # 
            # For the decomposition: sum_r U[a*m+b, r] * V[c*p+d, r] * W[d*n+a, r] = T[...]
            # The target tensor has T = 1 when b == c, 0 otherwise.
            
            # Build target tensor (sparse)
            # T[i, j, k] where i = a*m+b, j = c*p+d, k = d*n+a
            # T[i,j,k] = 1 if b == c
            
            # Compute actual tensor from decomposition
            # This is expensive for large R, so we compute residual in a smarter way
            # Residual = ||sum_r U[:,r] V[:,r]^T W[:,r]^T - T||_F^2
            # = sum_r sum_s (U^T V)_{rs} (U^T W)_{rs} ... no
            
            # Better: compute the tensor as sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
            # and compare to T. But this is O(nm*mp*pn*R) which is too slow.
            
            # Alternative: compute residual as sum over all (i,j,k) of (sum_r U[i,r]V[j,r]W[k,r] - T[i,j,k])^2
            # This is also expensive.
            
            # Use a random subset of entries for speed, or use the fact that T is sparse.
            # T[i,j,k] is nonzero only when b == c, i.e., when (i % m) == (j // p)
            # And k must equal (j % p) * n + (i // m)
            
            # Let's compute residual using a random sample for speed
            # Actually, for exact verification we need all entries. But for optimization,
            # a random sample might suffice to guide the search.
            
            # For now, compute on a random subset
            n_samples = min(5000, nm * mp * pn)
            idx_i = np.random.randint(0, nm, n_samples)
            idx_j = np.random.randint(0, mp, n_samples)
            idx_k = np.random.randint(0, pn, n_samples)
            
            # Compute sum_r U[i,r] V[j,r] W[k,r] for each sample
            # This is U[idx_i, :] * V[idx_j, :] * W[idx_k, :] summed over r
            vals = np.einsum('ir,jr,kr->ij', U[idx_i, :], V[idx_j, :], W[idx_k, :])
            
            # Target values
            targets = np.zeros(n_samples)
            for s in range(n_samples):
                i, j, k = idx_i[s], idx_j[s], idx_k[s]
                a, b = divmod(i, m)
                c, d = divmod(j, p)
                if b == c and k == d * n + a:
                    targets[s] = 1
            
            res = np.sum((vals - targets) ** 2)
            return res
        
        def get_params():
            return U, V, W
        
        return U, V, W, residual, get_params
    
    # Try to find lower rank using iterative refinement
    # Start from schoolbook and try to reduce rank
    
    # First, let's try a simple approach: for ranks slightly below schoolbook,
    # use random initialization and a few gradient steps
    
    def try_rank(R, n_restarts=5, max_iter=100):
        """Try to find rank R decomposition."""
        best_res = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Simple gradient descent
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Compute residual and gradients
                # Using random sampling for speed
                n_samples = 2000
                idx_i = np.random.randint(0, nm, n_samples)
                idx_j = np.random.randint(0, mp, n_samples)
                idx_k = np.random.randint(0, pn, n_samples)
                
                # Forward
                UV = U[idx_i, :] * V[idx_j, :]  # (n_samples, R)
                vals = np.sum(UV * W[idx_k, :], axis=1)  # (n_samples,)
                
                # Targets
                targets = np.zeros(n_samples)
                for s in range(n_samples):
                    i, j, k = idx_i[s], idx_j[s], idx_k[s]
                    a, b = divmod(i, m)
                    c, d = divmod(j, p)
                    if b == c and k == d * n + a:
                        targets[s] = 1
                
                # Residual
                res = np.sum((vals - targets) ** 2)
                
                if res < best_res:
                    best_res = res
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if res < 1e-10:
                    break
                
                # Gradients
                # d/dU[i,r] = 2 * sum_s (vals[s] - targets[s]) * V[j_s, r] * W[k_s, r]
                # This is expensive to compute exactly. Use approximation.
                err = vals - targets  # (n_samples,)
                
                # dU[i,r] = 2 * sum_s err[s] * V[j_s, r] * W[k_s, r] * (1 if i == i_s else 0)
                # = 2 * err[s] * V[j_s, r] * W[k_s, r] for s where i_s == i
                
                # This is still expensive. Let's use a simpler heuristic.
                # Just do random perturbation and accept if it improves.
                
                # Random perturbation
                dU = np.random.randn(nm, R) * lr
                dV = np.random.randn(mp, R) * lr
                dW = np.random.randn(pn, R) * lr
                
                U_new = U + dU
                V_new = V + dV
                W_new = W + dW
                
                # Check new residual
                UV_new = U_new[idx_i, :] * V_new[idx_j, :]
                vals_new = np.sum(UV_new * W_new[idx_k, :], axis=1)
                res_new = np.sum((vals_new - targets) ** 2)
                
                if res_new < res:
                    U, V, W = U_new, V_new, W_new
                else:
                    # Try smaller step
                    lr *= 0.5
                    if lr < 1e-6:
                        break
        
        return best_res, best_params
    
    # Try ranks from schoolbook - 1 down to a minimum
    min_rank_to_try = max(40, rank_sb - 20)
    
    for R in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        res, params = try_rank(R, n_restarts=3, max_iter=50)
        
        if res < 1e-8 and params is not None:
            U, V, W = params
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break
    
    # Verify the best solution
    if best_rank < rank_sb:
        # Full verification
        U = best_U
        V = best_V
        W = best_W
        
        # Check all entries
        max_err = 0
        for i in range(nm):
            a, b = divmod(i, m)
            for j in range(mp):
                c, d = divmod(j, p)
                if b != c:
                    continue
                k = d * n + a
                val = np.dot(U[i, :], V[j, :] * W[k, :])
                if abs(val - 1) > max_err:
                    max_err = abs(val - 1)
        
        # Also check that entries where b != c are close to 0
        for i in range(nm):
            a, b = divmod(i, m)
            for j in range(mp):
                c, d = divmod(j, p)
                if b == c:
                    continue
                k = d * n + a  # This k won't be the "correct" one, but we still need to check
                val = np.dot(U[i, :], V[j, :] * W[k, :])
                if abs(val) > max_err:
                    max_err = abs(val)
        
        if max_err > 1e-6:
            # Solution is not exact enough, fall back to schoolbook
            best_rank = rank_sb
            best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
