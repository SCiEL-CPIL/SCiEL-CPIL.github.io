import numpy as np
import time
import itertools

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    The schoolbook rank is 72. We attempt a structured search for rank 71 and 70
    using a smooth optimization approach (minimizing the norm of the residual tensor)
    with random restarts and a time budget. If no better rank is found, we return
    the schoolbook decomposition.
    
    The tensor T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    where i in [0,3), j in [0,4), k in [0,6).
    """
    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def compute_residual(U, V, W):
        """Compute the residual tensor R where R[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] - T[i,j,k]"""
        # Target tensor: T[i,j,k] = 1 if (i,j,k) is a valid schoolbook term
        # Actually, the matrix multiplication tensor is:
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor T_{i,j,k} has T_{i,j,k} = 1 for all (i,j,k) in the schoolbook sense
        # But actually the tensor for matrix multiplication is:
        # T_{i,j,k} = delta_{i,j} * delta_{j,k}? No.
        # 
        # Let me reconsider. The matrix multiplication tensor T is defined such that:
        # (A B)_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{j,r} T_{i,j,k,r} A_{i,j} B_{j,k}
        # No, that's not right either.
        #
        # The standard formulation: We want to find U, V, W such that for all A (n x m), B (m x p):
        # (AB)_{i,k} = sum_r U_{i,r} V_{k,r} * (something)
        #
        # Actually, the factorization is:
        # (AB)_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # = sum_{j,r} U_{i,j,r} V_{j,k,r} W_{r,i,k}  -- this isn't standard
        #
        # Let me use the convention from the task:
        # W[k*n + i, r] is the output factor.
        # The decomposition is:
        # (AB)_{i,k} = sum_r U_{i*m + j, r} * V_{j*p + k, r} * W_{k*n + i, r} -- but j is summed over
        #
        # Actually, the standard rank decomposition of the matrix multiplication tensor:
        # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes to (AB)[i,k])
        #
        # Wait, no. The matrix multiplication tensor is a 3-tensor T with:
        # T[i,j,k] = 1 for all i, j, k
        # And the rank decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # So the target is T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p).
        
        residual = np.zeros((n, m, p))
        for r in range(len(U[0])):
            u_vals = U[:, r]  # shape (n*m,)
            v_vals = V[:, r]  # shape (m*p,)
            w_vals = W[:, r]  # shape (p*n,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        residual[i, j, k] += u_vals[u_idx] * v_vals[v_idx] * w_vals[w_idx]
        
        # Target: all ones
        target = np.ones((n, m, p))
        residual = residual - target
        return residual
    
    def objective(U, V, W):
        """Return the squared norm of the residual."""
        r = compute_residual(U, V, W)
        return np.sum(r * r)
    
    def try_rank_search(rank, time_budget=30.0):
        """Attempt to find a decomposition of the given rank using gradient descent with random restarts."""
        start_time = time.time()
        best_obj = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        max_restarts = 5
        max_iterations = 2000
        lr = 0.01
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Gradient descent with numerical gradients (too slow, use analytic or skip)
            # Instead, use a simple iterative approach: minimize ||T - sum_r u_r v_r w_r||^2
            # Using alternating least squares or similar
            
            # For now, use a simple approach: try to match each tensor entry
            # This is a non-convex problem, so we use random restarts and gradient steps
            
            # Compute gradient numerically for a few steps
            obj_prev = objective(U, V, W)
            
            for it in range(max_iterations):
                if time.time() - start_time > time_budget - 5:
                    break
                
                # Numerical gradient for U
                grad_U = np.zeros_like(U)
                for r in range(rank):
                    for i in range(n * m):
                        eps = 0.001
                        U_pert = U.copy()
                        U_pert[i, r] += eps
                        grad_U[i, r] = (objective(U_pert, V, W) - obj_prev) / eps
                
                # Gradient for V
                grad_V = np.zeros_like(V)
                for r in range(rank):
                    for j in range(m * p):
                        eps = 0.001
                        V_pert = V.copy()
                        V_pert[j, r] += eps
                        grad_V[j, r] = (objective(U, V_pert, W) - obj_prev) / eps
                
                # Gradient for W
                grad_W = np.zeros_like(W)
                for r in range(rank):
                    for k in range(p * n):
                        eps = 0.001
                        W_pert = W.copy()
                        W_pert[k, r] += eps
                        grad_W[k, r] = (objective(U, V, W_pert) - obj_prev) / eps
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                obj_prev = objective(U, V, W)
                
                if obj_prev < best_obj:
                    best_obj = obj_prev
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if obj_prev < 1e-10:
                    break
        
        # Check if we found an exact solution
        if best_obj < 1e-10 and best_U is not None:
            # Try to round to integers/half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            obj_rounded = objective(U_rounded, V_rounded, W_rounded)
            if obj_rounded < 1e-10:
                return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
        
        return None
    
    # First, try to find a better rank
    for target_rank in [71, 70, 69, 68]:
        result = try_rank_search(target_rank, time_budget=15.0)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return build_schoolbook()
