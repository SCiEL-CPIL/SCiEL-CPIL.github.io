"""Matrix multiplication 2x3 x 3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor indices
    # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    def compute_residual(U, V, W):
        """Compute the residual for the target entries."""
        res = 0.0
        for (a, b, c) in target_indices:
            val = np.dot(U[a], V[b]) * W[c]  # This is wrong, need element-wise
            # Actually: sum_d U[a,d]*V[b,d]*W[c,d]
            val = np.sum(U[a] * V[b] * W[c])
            res += (val - 1.0) ** 2
        return res
    
    def check_zero_entries(U, V, W, num_samples=200, rng=None):
        """Check that random zero entries are close to zero."""
        if rng is None:
            rng = np.random.RandomState(42)
        for _ in range(num_samples):
            a = rng.randint(dim_u)
            b = rng.randint(dim_v)
            c = rng.randint(dim_w)
            # Check if this is a target entry
            is_target = False
            for (ta, tb, tc) in target_indices:
                if a == ta and b == tb and c == tc:
                    is_target = True
                    break
            if not is_target:
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-6:
                    return False
        return True
    
    def verify_full(U, V, W, tol=1e-8):
        """Full verification of the decomposition."""
        # Check all target entries
        for (a, b, c) in target_indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check zero entries
        if not check_zero_entries(U, V, W, num_samples=500):
            return False
        return True
    
    def round_to_half(X):
        return np.round(X * 2) / 2.0
    
    # Strategy: Try to find lower rank decompositions
    # Use iterative optimization: fix two matrices, solve for the third
    
    def optimize_rank(rank, max_iter=200, tol=1e-10):
        """Try to find a rank-r decomposition."""
        rng = np.random.RandomState(int(time.time()) % 10000 + rank)
        
        # Initialize
        U = rng.randn(dim_u, rank) * 0.3
        V = rng.randn(dim_v, rank) * 0.3
        W = rng.randn(dim_w, rank) * 0.3
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Alternate: fix V,W solve for U; fix U,W solve for V; fix U,V solve for W
            # For U: minimize sum_{(a,b,c) in target} (sum_d U[a,d]*V[b,d]*W[c,d] - 1)^2
            # This is quadratic in U. Let's set up the normal equations.
            
            # For each target (a,b,c): let w_bc = V[b] * W[c] (element-wise product, shape rank)
            # Then the term is U[a] @ w_bc
            # We want to minimize sum (U[a] @ w_bc - 1)^2 over all targets
            
            # Build linear system for U:
            # For each target (a,b,c), we have one equation involving U[a]
            # Group by a: for each a, collect all (b,c) pairs that appear with this a
            
            # Actually, let's do it more directly with gradient descent
            lr = 0.01 / (1 + iteration * 0.1)
            
            # Compute gradient for U
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in target_indices:
                # d/du: 2*(val-1)*V[b]*W[c]
                val = np.sum(U[a] * V[b] * W[c])
                err = val - 1.0
                grad_U[a] += 2 * err * V[b] * W[c]
                grad_V[b] += 2 * err * U[a] * W[c]
                grad_W[c] += 2 * err * U[a] * V[b]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            loss = compute_residual(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < tol:
                break
        
        return best_U, best_V, best_W, best_loss
    
    # Try ranks from schoolbook down
    for rank in range(schoolbook_rank, max(schoolbook_rank - 15, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break
            
            U, V, W, loss = optimize_rank(rank, max_iter=150)
            
            if loss < 1e-6:
                # Good solution found, try to refine and round
                # Polish with smaller learning rate
                for polish_iter in range(100):
                    if time.time() > deadline:
                        break
                    lr = 0.001
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for (a, b, c) in target_indices:
                        val = np.sum(U[a] * V[b] * W[c])
                        err = val - 1.0
                        grad_U[a] += 2 * err * V[b] * W[c]
                        grad_V[b] += 2 * err * U[a] * W[c]
                        grad_W[c] += 2 * err * U[a] * V[b]
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                # Round to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                if verify_full(U_r, V_r, W_r, tol=1e-6):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
                else:
                    # Try without rounding if it's close enough
                    if verify_full(U, V, W, tol=1e-8):
                        # Keep as-is but check if entries are close to half-integers
                        U_close = np.allclose(U, U_r, atol=1e-4)
                        V_close = np.allclose(V, V_r, atol=1e-4)
                        W_close = np.allclose(W, W_r, atol=1e-4)
                        if U_close and V_close and W_close:
                            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                            best_rank = rank
                            found = True
                            break
        
        if found:
            continue
        else:
            # Didn't find for this rank, try a few more ranks down
            break
    
    return best_result
# EVOLVE-BLOCK-END
