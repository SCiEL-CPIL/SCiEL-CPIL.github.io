"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the schoolbook tensor
    def tensor_entry(i, j, k):
        return 1
    
    # We want to find a rank-r decomposition.
    # The tensor T is a 3-tensor of dimensions (n*m) x (m*p) x (p*n) ? No.
    # The matrix multiplication tensor T_{ijk} is such that:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is indexed by (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # But the interface requires U, V, W matrices of specific shapes.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    # The decomposition is: sum_r U_{\alpha, r} V_{\beta, r} W_{\gamma, r} = T_{\alpha, \beta, \gamma}
    # where \alpha = i*m + j, \beta = j*p + k, \gamma = k*n + i.
    
    # Let's define the target tensor T as a dense array of shape (n*m, m*p, p*n).
    # T[\alpha, \beta, \gamma] = 1 if there exist i, j, k such that:
    # \alpha = i*m + j, \beta = j*p + k, \gamma = k*n + i, and the product is 1.
    # Actually, the standard tensor for matmul is T(i,j,k) = 1.
    # The mapping to the interface is:
    # Row index of U: i*m + j
    # Row index of V: j*p + k
    # Row index of W: k*n + i
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Try to find a rank-71 decomposition.
    # Since n=3, m=4, p=6, n*m*p = 72.
    # It is known that the rank of 3x4x6 matmul is 71.
    # We will use a random initialization and minimize the Frobenius norm of the residual.
    # Residual R = T - sum_r U_r \otimes V_r \otimes W_r
    # We can vectorize this.
    # Let U be (n*m, r), V be (m*p, r), W be (p*n, r).
    # The reconstructed tensor \hat{T} is formed by:
    # \hat{T}[\alpha, \beta, \gamma] = sum_r U[\alpha, r] * V[\beta, r] * W[\gamma, r]
    
    # We can flatten U, V, W into a single vector x.
    # However, the objective is non-convex and non-linear.
    # A better approach for small ranks is to use a grid search or random restarts with gradient descent on the parameters.
    
    # Let's try a simple random restart approach with L-BFGS-B.
    # Time limit is 120s. We should use a fraction of that.
    
    deadline = time.time() + 110  # 110 seconds limit
    
    def build_decomposition(U_flat, V_flat, W_flat, r):
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        # Reconstruct tensor
        # T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        # But this is slow.
        # Alternative: for each r, add U[:,r][:,None,None] * V[:,r][None,:,None] * W[:,r][None,None,:] to T_hat.
        T_hat = np.zeros((n*m, m*p, p*n))
        for r_ in range(r):
            T_hat += U[:, r_][:, None, None] * V[:, r_][None, :, None] * W[:, r_][None, None, :]
        return U, V, W, T_hat

    def objective(x):
        r = len(x) // 3
        # x is [U_flat, V_flat, W_flat]
        U_flat = x[:r*(n*m)]
        V_flat = x[r*(n*m):2*r*(n*m)]
        W_flat = x[2*r*(n*m):]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual
        R = T - T_hat
        # Frobenius norm squared
        return np.sum(R * R)

    # We want to find a solution with integer/half-integer entries.
    # The random search will find a real solution.
    # If the rank is truly 71, there exists a solution with rational entries.
    # We can try to round the solution to half-integers and check if it is exact.
    
    def check_decomposition(U, V, W, tol=1e-6):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - T_hat
        return np.max(np.abs(R)) < tol

    best_result = None
    best_rank = rank_schoolbook
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    # Initialize schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                best_U[i*m+j, c] = 1
                best_V[j*p+k, c] = 1
                best_W[k*n+i, c] = 1

    # Try to find rank 71
    target_rank = 71
    if target_rank < best_rank:
        n_vars = target_rank * (n*m + m*p + p*n)
        # n*m = 12, m*p = 24, p*n = 18. Sum = 54.
        # 71 * 54 = 3834 variables.
        # This is a large optimization problem.
        # L-BFGS-B might be slow or get stuck.
        # Let's try a few random restarts.
        
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Initialize with small random values, or maybe initialize from schoolbook and drop one column?
            # Dropping one column from schoolbook (rank 72) gives a rank 71 approximation with residual 1.
            # Let's try initializing from schoolbook and perturbing.
            
            U_init = best_U[:, :target_rank].copy()
            V_init = best_V[:, :target_rank].copy()
            W_init = best_W[:, :target_rank].copy()
            
            # Perturb
            U_init += np.random.randn(n*m, target_rank) * 0.1
            V_init += np.random.randn(m*p, target_rank) * 0.1
            W_init += np.random.randn(p*n, target_rank) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                if res.success:
                    x_opt = res.x
                    U_opt = x_opt[:target_rank*(n*m)].reshape(n*m, target_rank)
                    V_opt = x_opt[target_rank*(n*m):2*target_rank*(n*m)].reshape(m*p, target_rank)
                    W_opt = x_opt[2*target_rank*(n*m):].reshape(p*n, target_rank)
                    
                    # Check if exact (within tolerance)
                    if check_decomposition(U_opt, V_opt, W_opt, tol=1e-5):
                        # Try to round to half-integers
                        # The problem requires integer/half-integer entries.
                        # If the solution is close to such values, we can use them.
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        if check_decomposition(U_round, V_round, W_round, tol=1e-6):
                            best_rank = target_rank
                            best_U = U_round
                            best_V = V_round
                            best_W = W_round
                            break # Found a solution
            except Exception as e:
                pass
            
            if time.time() > deadline:
                break

    # Convert to lists for output
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
