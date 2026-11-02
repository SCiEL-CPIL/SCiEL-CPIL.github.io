"""Matrix multiplication 2x3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(decomp, n, m, p):
    """Verify a decomposition exactly reproduces the matrix multiplication tensor."""
    size_a = n * m
    size_b = m * p
    size_c = p * n
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    # Build target
    target = np.zeros((size_a, size_b, size_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition
    if U.size == 0 or V.size == 0 or W.size == 0:
        return False
    R = U.shape[1] if U.ndim == 2 else 1
    if U.shape != (size_a, R) or V.shape != (size_b, R) or W.shape != (size_c, R):
        return False
    computed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(computed, target, atol=1e-12, rtol=1e-12)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    size_a = n * m  # 6
    size_b = m * p  # 21
    size_c = p * n  # 14
    rank_schoolbook = n * m * p  # 42

    # Schoolbook decomposition as fallback
    U_school = np.zeros((size_a, rank_schoolbook), dtype=int)
    V_school = np.zeros((size_b, rank_schoolbook), dtype=int)
    W_school = np.zeros((size_c, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    fallback = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build target tensor
    target_tensor = np.zeros((size_a, size_b, size_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i * m + j, j * p + k, k * n + i] = 1.0

    # Structural decomposition: split inner dimension 3 into 2+1
    # Use Strassen-like for 2x2*2x7 block + schoolbook for 2x1*1x7
    
    # For 2x2*2x7 (indices: A rows 0,1 cols 0,1; B rows 0,1 cols 0..6; C rows 0,1 cols 0..6):
    # We can use the known rank-28 schoolbook, but let's try a structural improvement.
    
    # Actually, let me try a direct constructive approach for specific ranks.
    
    # Strategy: Try rank 21 (which is half of schoolbook 42, or 7*3 = p*m)
    # and rank 14, etc., using optimization.
    
    # For small problems like 2x3x7, let's try structured search.
    
    # Key insight: The tensor M_{2,3,7} has structure. Let's try to find
    # a decomposition using the fact that one dimension is small (n=2).
    
    # Each output C[k,i] = sum_j A[i,j] * B[j,k]
    # With n=2, we have C[k,0] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #            C[k,1] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # For each k (0..6), we need to compute 2 outputs, each a sum of 3 products.
    # Total: 7 * 2 * 3 = 42 schoolbook terms.
    
    # Can we share computations across different k values?
    # The A entries are shared across all k, B entries are specific to k.
    # So for fixed A, we need to compute 14 outputs (7 k's * 2 i's) from 21 B entries.
    
    # Let me try to find a rank-R decomposition via optimization.
    
    def try_optimize(R, num_restarts=3, max_iter=200):
        """Try to find rank-R decomposition using gradient-based optimization."""
        if time.time() > deadline:
            return None
        
        total_params = (size_a + size_b + size_c) * R
        best_loss = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Initialize from schoolbook (take first R terms)
                params = np.zeros(total_params)
                for r in range(min(R, rank_schoolbook)):
                    i = r // (m * p)
                    rem = r % (m * p)
                    j = rem // p
                    k = rem % p
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    params[a * R + r] = 1.0
                    params[(size_a + b) * R + r] = 1.0
                    params[(size_a + size_b + c) * R + r] = 1.0
                # Add small noise to break symmetry
                params += np.random.normal(0, 0.01, total_params)
            elif restart == 1:
                # Sparse random initialization
                params = np.zeros(total_params)
                for _ in range(R * 4):
                    idx = np.random.randint(0, total_params)
                    params[idx] = np.random.choice([-1.0, 1.0]) * np.random.choice([0.5, 1.0])
            else:
                # Gaussian random
                params = np.random.normal(0, 0.5, total_params)
            
            def loss_and_grad(x):
                U = x[:size_a * R].reshape((size_a, R))
                V = x[size_a * R:(size_a + size_b) * R].reshape((size_b, R))
                W = x[(size_a + size_b) * R:].reshape((size_c, R))
                computed = np.einsum('ar,br,cr->abc', U, V, W)
                diff = computed - target_tensor
                loss = np.sum(diff * diff)
                
                # Gradients
                tmp1 = np.einsum('abc,br->acr', diff, V)
                grad_U = 2 * np.einsum('acr,cr->ar', tmp1, W)
                
                tmp2 = np.einsum('abc,ar->bcr', diff, U)
                grad_V = 2 * np.einsum('bcr,cr->br', tmp2, W)
                
                grad_W = 2 * np.einsum('bcr,br->cr', tmp2, V)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad
            
            # Simple gradient descent with adaptive learning rate
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                loss, grad = loss_and_grad(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                if loss < 1e-8:
                    break
                params -= lr * grad
                # Reduce learning rate slowly
                lr *= 0.995
            
            if best_loss < 1e-8:
                break
        
        if best_params is None or best_loss > 1e-6:
            return None
        
        # Try to snap to half-integers
        U_flat = best_params[:size_a * R]
        V_flat = best_params[size_a * R:(size_a + size_b) * R]
        W_flat = best_params[(size_a + size_b) * R:]
        
        for factor in [1, 2, 4, 8]:
            U_snap = np.round(U_flat * factor) / factor
            V_snap = np.round(V_flat * factor) / factor
            W_snap = np.round(W_flat * factor) / factor
            U_s = U_snap.reshape((size_a, R))
            V_s = V_snap.reshape((size_b, R))
            W_s = W_snap.reshape((size_c, R))
            computed_snap = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
            if np.max(np.abs(computed_snap - target_tensor)) < 1e-9:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        # If very close, accept floating point
        if best_loss < 1e-10:
            U_s = best_params[:size_a * R].reshape((size_a, R))
            V_s = best_params[size_a * R:(size_a + size_b) * R].reshape((size_b, R))
            W_s = best_params[(size_a + size_b) * R:].reshape((size_c, R))
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None

    # Try various ranks
    best_result = fallback
    for R in [25, 30, 35, 36, 38, 39, 40, 41, 42]:
        if time.time() > deadline:
            break
        result = try_optimize(R, num_restarts=2, max_iter=100)
        if result is not None and _verify(result, n, m, p):
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
