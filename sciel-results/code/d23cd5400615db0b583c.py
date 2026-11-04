"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS

    def make_trivial():
        """Create the trivial rank n*m*p decomposition."""
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
        return U, V, W, rank

    U_triv, V_triv, W_triv, trivial_rank = make_trivial()
    best_result = {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    best_rank = trivial_rank

    def residual(U, V, W, rank):
        """Compute the residual vector for the tensor decomposition."""
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Residual = 1 - sum_r ...
        res = np.ones((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[i, j, k] = 1.0 - s
        return res

    def residual_flat(U, V, W, rank):
        """Compute flat residual."""
        res = np.ones(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :rank], V[j * p + k, :rank] * W[k * n + i, :rank])
                    res[idx] = 1.0 - s
                    idx += 1
        return res

    def jacobian(U, V, W, rank):
        """Compute the Jacobian of the residual with respect to U, V, W parameters."""
        # Parameters: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        # Total params: (n*m + m*p + p*n) * rank
        num_params = (n * m + m * p + p * n) * rank
        num_res = n * m * p
        J = np.zeros((num_res, num_params))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res_idx = i * m * p + j * p + k
                    # d/dU[i*m+j, r] of (1 - sum_r U*V*W) = -V[j*p+k, r] * W[k*n+i, r]
                    for r in range(rank):
                        param_idx = (i * m + j) * rank + r
                        J[res_idx, param_idx] = -V[j * p + k, r] * W[k * n + i, r]
                    # d/dV[j*p+k, r] = -U[i*m+j, r] * W[k*n+i, r]
                    for r in range(rank):
                        param_idx = (n * m) * rank + (j * p + k) * rank + r
                        J[res_idx, param_idx] = -U[i * m + j, r] * W[k * n + i, r]
                    # d/dW[k*n+i, r] = -U[i*m+j, r] * V[j*p+k, r]
                    for r in range(rank):
                        param_idx = (n * m + m * p) * rank + (k * n + i) * rank + r
                        J[res_idx, param_idx] = -U[i * m + j, r] * V[j * p + k, r]
        return J

    def try_rank(rank, num_restarts=10, max_newton_iters=50, seed_base=0):
        """Try to find a decomposition of given rank using Newton's method."""
        rng = np.random.RandomState(seed_base)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            
            # Scale to get reasonable magnitudes
            U *= 1.0 / np.sqrt(rank)
            V *= 1.0 / np.sqrt(rank)
            W *= 1.0 / np.sqrt(rank)
            
            for it in range(max_newton_iters):
                if time.time() > deadline - 3:
                    break
                
                # Compute residual
                res = residual_flat(U, V, W, rank)
                max_res = np.max(np.abs(res))
                
                if max_res < 1e-10:
                    # Found a solution!
                    return U, V, W, rank
                
                # Compute Jacobian
                J = jacobian(U, V, W, rank)
                
                # Solve J * delta = -res for delta (least squares)
                try:
                    delta, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
                except:
                    break
                
                # Damping
                step = 1.0
                for _ in range(10):
                    # Apply step
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    
                    U_flat = U.flatten()
                    V_flat = V.flatten()
                    W_flat = W.flatten()
                    
                    U_flat_new = U_flat + step * delta[:n * m * rank]
                    V_flat_new = V_flat + step * delta[n * m * rank:(n * m + m * p) * rank]
                    W_flat_new = W_flat + step * delta[(n * m + m * p) * rank:]
                    
                    U_new = U_flat_new.reshape(n * m, rank)
                    V_new = V_flat_new.reshape(m * p, rank)
                    W_new = W_flat_new.reshape(p * n, rank)
                    
                    res_new = residual_flat(U_new, V_new, W_new, rank)
                    if np.max(np.abs(res_new)) < max_res:
                        U, V, W = U_new, V_new, W_new
                        break
                    step *= 0.5
                
                if step < 1e-10:
                    break
            
            # Final check
            res = residual_flat(U, V, W, rank)
            if np.max(np.abs(res)) < 1e-9:
                return U, V, W, rank
        
        return None

    # Try various ranks from lower to upper
    # Lower bound is max(n*p, m*p, n*m) = 28
    # Known results: rank of M_{3,4,7} is known to be at most some value
    # Let's try ranks from 40 up to 83
    
    for rank in range(40, trivial_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=5, max_newton_iters=30, seed_base=rank * 100)
        
        if result is not None:
            U, V, W, r = result
            # Verify
            res = residual_flat(U, V, W, r)
            if np.max(np.abs(res)) < 1e-9:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found a valid decomposition, return it
    
    return best_result
# EVOLVE-BLOCK-END
