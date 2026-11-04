"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank(n, m, p, target_rank, max_time=10.0, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using random restarts + local optimization.
    Returns (U, V, W) as lists if found, else None.
    """
    t0 = time.time()
    np.random.seed(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build the tensor T as a 3D array T[i,j,k] = 1
    # We want U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sum over c = 1
    
    # Reshape approach: 
    # Let U be (n*m, r), V be (m*p, r), W be (p*n, r)
    # We can think of this as: for each c, the outer product structure
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # This is equivalent to: 
    # For fixed j, the matrix U_j (n x r) * V_j (r x p) * W_j (p x n) ... no that's not right.
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c]
    
    # Let's define for each c: 
    #   A_c is n x p matrix where A_c[i,k] = U[i*m+j, c] * W[k*n+i, c] ... but this depends on j
    #   That's not quite separable in the standard way.
    
    # Better: think of it as a tensor decomposition where:
    # T is a 3-tensor of size (n*m) x (m*p) x (p*n) but with specific index mapping.
    # Actually, the standard matrix multiplication tensor is T_{n,m,p} with indices (i,j,k) 
    # and T[i,j,k] = 1. The rank is the minimum r such that:
    # T[i,j,k] = sum_{c=1}^r U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where U_c is n x m, V_c is m x p, W_c is p x n.
    
    # So we have r matrices U_c (n x m), V_c (m x p), W_c (p x n) such that
    # for all i, j, k: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # Let's flatten: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # Objective: minimize sum_{i,j,k} (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1)^2
    
    def _objective_and_grad(params):
        # params: flattened U, V, W of length (n*m + m*p + p*n) * r
        U_flat = params[:n_m * r]
        V_flat = params[n_m * r : n_m * r + m_p * r]
        W_flat = params[n_m * r + m_p * r : n_m * r + m_p * r + p_n * r]
        
        U = U_flat.reshape(n_m, r)
        V = V_flat.reshape(m_p, r)
        W = W_flat.reshape(p_n, r)
        
        # Compute the tensor and compare to 1
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can vectorize this.
        
        # Create index arrays
        # Let's compute the sum for all (i,j,k)
        # T_computed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For efficiency, loop over c and accumulate
        # Or use einsum-like approach
        
        # Let's do it by iterating over c and building up
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We can write this as: for each c, the contribution is
        # (U_c reshaped to n x m) * (V_c reshaped to m x p) * (W_c reshaped to p x n)
        # but multiplied element-wise in a specific way.
        
        # Actually: U[i*m+j, c] is the (i,j) entry of matrix U_c (n x m)
        # V[j*p+k, c] is the (j,k) entry of matrix V_c (m x p)  
        # W[k*n+i, c] is the (k,i) entry of matrix W_c (p x n)
        
        # So T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # Let's compute this efficiently
        # For each c:
        #   contrib_c[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        #   T[i,j,k] = sum_c contrib_c[i,j,k]
        
        # We can compute this using matrix operations:
        # For fixed c: 
        #   Let A = U_c (n x m), B = V_c (m x p), C = W_c (p x n)
        #   contrib[i,j,k] = A[i,j] * B[j,k] * C[k,i]
        
        # This is not a standard matrix product. Let's just compute it directly.
        
        # For performance, let's precompute the index mappings
        # i ranges 0..n-1, j ranges 0..m-1, k ranges 0..p-1
        
        # Let's build the tensor
        # T = np.zeros((n, m, p))
        # for c in range(r):
        #     Uc = U[:, c].reshape(n, m)
        #     Vc = V[:, c].reshape(m, p)
        #     Wc = W[:, c].reshape(p, n)
        #     for i in range(n):
        #         for j in range(m):
        #             for k in range(p):
        #                 T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        
        # This is O(r*n*m*p) which for n=3,m=4,p=7,r~20 is 3*4*7*20 = 1680, very fast.
        
        T = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # contrib[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
        
        error = T - 1.0
        obj = np.sum(error ** 2)
        
        # Gradient: 
        # d(obj)/d(U[i*m+j, c]) = 2 * sum_{i,j,k} error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # Let's compute this
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    # dU[i,j,c] = 2 * sum_k error[i,j,k] * Vc[j,k] * Wc[k,i]
                    grad_U[i * m + j, c] = 2.0 * np.dot(error[i, j, :], Vc[j, :]) * Wc[:, i]
                    # Actually: sum_k error[i,j,k] * Vc[j,k] * Wc[k,i]
                    # = sum_k error[i,j,k] * Vc[j,k] * Wc[k,i]
                    # Let me recompute:
                    # grad_U[i*m+j, c] = 2 * sum_k error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                    # = 2 * sum_k error[i,j,k] * Vc[j,k] * Wc[k,i]
                    # This is 2 * (error[i,j,:] * Vc[j,:]) @ Wc[:,i] ... no
                    # = 2 * sum_k error[i,j,k] * Vc[j,k] * Wc[k,i]
                    # Let's compute it as:
                    # temp = error[i,j,:] * Vc[j,:]  (vector of length p)
                    # grad_U[i*m+j, c] = 2 * dot(temp, Wc[:, i])
                    # But I already computed it above incorrectly. Let me redo.
        
        # Let me recompute gradients properly
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = error[i, j, k]
                        grad_U[i * m + j, c] += 2.0 * e * Vc[j, k] * Wc[k, i]
                        grad_V[j * p + k, c] += 2.0 * e * Uc[i, j] * Wc[k, i]
                        grad_W[k * n + i, c] += 2.0 * e * Uc[i, j] * Vc[j, k]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad
    
    r = target_rank
    best_obj = np.inf
    best_params = None
    max_restarts = 20
    restarts = 0
    
    while time.time() - t0 < max_time and restarts < max_restarts:
        restarts += 1
        # Random initialization
        params = np.random.randn((n_m + m_p + p_n) * r) * 0.1
        
        # Use L-BFGS-B
        result = minimize(_objective_and_grad, params, method='L-BFGS-B', 
                         jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
        
        if best_obj < 1e-6:
            break
    
    if best_params is not None and best_obj < 1e-4:
        U_flat = best_params[:n_m * r]
        V_flat = best_params[n_m * r : n_m * r + m_p * r]
        W_flat = best_params[n_m * r + m_p * r : n_m * r + m_p * r + p_n * r]
        U = U_flat.reshape(n_m, r)
        V = V_flat.reshape(m_p, r)
        W = W_flat.reshape(p_n, r)
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Theoretical lower bounds: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also n*m*p = 84 is the schoolbook rank
    # Known: rank(M_{3,4,4}) = 18 (Strassen-like improvements)
    # For M_{3,4,7}, we can try ranks from 28 upward
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to improve
    # Lower bound is max(n*m, m*p, p*n) = 28
    # Try ranks from 28 to best_rank - 1
    for rank in range(28, best_rank):
        if time.time() - t0 > _DEADLINE - 30:  # leave 30s margin
            break
        
        result = _try_rank(n, m, p, rank, max_time=8.0, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            break  # Found a valid decomposition, use it
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

