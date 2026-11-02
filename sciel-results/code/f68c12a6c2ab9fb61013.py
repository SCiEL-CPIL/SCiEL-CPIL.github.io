import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank to try, starting from 116 (known bound) down to 110
    # If no solution found, fall back to schoolbook rank 120
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Pre-compute the target tensor T
    # T[i, j, k] = 1 if it's a valid index, else 0
    # But the decomposition is defined by U, V, W such that
    # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,j,k} (for the basis tensor)
    # Actually, the tensor for matrix multiplication is T[i, j, k] = 1 for all i, j, k in the respective ranges.
    # Wait, the standard tensor for C = AB is T_{i, k, j} = 1? 
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # And U is (n*m, rank), V is (m*p, rank).
    # The tensor is T_{i, j, k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition is: T_{i, j, k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    
    target_T = np.ones((n, m, p))

    def check_solution(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # Reconstruct T
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T_recon[i, j, k] = s
        # Check if T_recon matches target_T (all ones)
        if np.allclose(T_recon, target_T, atol=1e-9):
            return True
        return False

    def round_to_half_integer(arr):
        return np.round(arr * 2.0) / 2.0

    def try_find_decomposition(rank):
        # Initialize U, V, W
        # Strategy: Start with schoolbook and merge terms randomly or use random init
        # Better: Use a structured init based on schoolbook terms
        U, V, W, _ = get_schoolbook()
        
        # If rank is 120, return it directly
        if rank == 120:
            return U, V, W, rank

        # For lower ranks, we need to find a valid decomposition.
        # This is a hard problem. We will use a randomized local search.
        # We start with a random set of rank-1 tensors and try to fit them.
        
        # However, a simpler heuristic: 
        # Try to find a solution by solving a linear system if we fix one matrix.
        # This is generally not possible for bilinear algorithms directly.
        
        # Let's try a specific known structure or a search.
        # Given the time limit, we will attempt a few random restarts with optimization.
        
        import scipy.optimize as opt
        
        def objective(params):
            # params: flattened U, V, W
            size_U = n * m * rank
            size_V = m * p * rank
            size_W = p * n * rank
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:size_U+size_V+size_W]
            
            U_m = U_flat.reshape(n*m, rank)
            V_m = V_flat.reshape(m*p, rank)
            W_m = W_flat.reshape(p*n, rank)
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_m[i*m+j, :] * V_m[j*p+k, :] * W_m[k*n+i, :])
                        err += (s - 1.0)**2
            return err

        # Initial guess: Use schoolbook terms, but we have fewer columns.
        # We can map the 120 schoolbook terms to 'rank' columns by averaging or random assignment.
        # A better init: Set U, V, W to random small values.
        
        best_error = float('inf')
        best_params = None
        
        num_restarts = 5 # Keep low to fit in time
        
        for r in range(num_restarts):
            # Random init
            np.random.seed(42 + r)
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            # Scale to ensure magnitude is reasonable
            # The product should be ~1. If U~0.1, V~0.1, W~0.1, product is 0.001.
            # We need product ~1. So scale by 10?
            # 10 * 10 * 10 = 1000. Too big.
            # Let's use 1.0 scale.
            U_init = np.random.randn(n*m, rank)
            V_init = np.random.randn(m*p, rank)
            W_init = np.random.randn(p*n, rank)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B or similar
            try:
                res = opt.minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 1000})
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
            except Exception:
                pass

        if best_params is not None and best_error < 1e-6:
            size_U = n * m * rank
            size_V = m * p * rank
            size_W = p * n * rank
            U_m = best_params[:size_U].reshape(n*m, rank)
            V_m = best_params[size_U:size_U+size_V].reshape(m*p, rank)
            W_m = best_params[size_U+size_V:size_U+size_V+size_W].reshape(p*n, rank)
            
            # Round to half integers
            U_rounded = round_to_half_integer(U_m)
            V_rounded = round_to_half_integer(V_m)
            W_rounded = round_to_half_integer(W_m)
            
            if check_solution(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded, rank

        return None

    # Try ranks from 116 down to 110
    # Note: 116 is the known integer bound.
    for r in range(116, 109, -1):
        start_time = time.time()
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W, rank = result
            # Convert to lists of lists as required by interface
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        if time.time() - start_time > 60: # Safety margin
            break
            
    # Fallback to schoolbook
    U, V, W, rank = get_schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
