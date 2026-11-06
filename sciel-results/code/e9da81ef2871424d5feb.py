import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    n = 24
    
    # Dense grid on unit circle for sup-norm evaluation
    # 8192 points is sufficient for 1e-9 accuracy with golden section refinement
    n_grid = 8192
    angles = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    z = np.exp(1j * angles)
    
    # Precompute the matrix Z where Z[j, k] = z_j^k
    # Z shape: (n_grid, n)
    Z = np.zeros((n_grid, n), dtype=np.complex128)
    Z[:, 0] = 1
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z
    
    def sup_norm(coeffs):
        """Compute max |P(z)| over |z|=1 using dense grid + golden-section refinement."""
        c = np.array(coeffs, dtype=np.complex128)
        # Evaluate P(z) = sum c_k z^k for all z on grid
        vals = Z @ c
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        
        # Golden-section refinement around the peak
        idx = np.argmax(abs_vals)
        dtheta = 2 * np.pi / n_grid
        a = angles[idx] - dtheta
        b = angles[idx] + dtheta
        
        # Helper for fast evaluation at a single point
        def f(theta):
            # P(theta) = sum c_k * exp(i * k * theta)
            # Re-evaluation is O(n) = 24 ops, which is very fast.
            k_vec = np.arange(n)
            return np.abs(np.sum(c * np.exp(1j * theta * k_vec)))
            
        gr = (np.sqrt(5) - 1) / 2
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = f(c1)
        f2 = f(c2)
        
        for _ in range(50):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = f(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = f(c2)
        
        refined = max(f1, f2)
        return max(m, refined)
    
    # Initial candidate: Rudin-Shapiro sequence
    def rudin_shapiro(length):
        p, q = [1], [1]
        while len(p) < length:
            p, q = p + q, p + [-c for c in q]
        return p[:length]
    
    best_coeffs = np.array(rudin_shapiro(n))
    best_val = sup_norm(best_coeffs)
    
    # Local search with vectorized steepest descent
    # For each coefficient i, flipping it changes P(z) by -2*c_i*z^i
    # We can compute the new sup norm for all possible single flips simultaneously
    
    def local_search_vectorized(coeffs, max_iters=10000):
        current = coeffs.copy()
        current_val = sup_norm(current)
        c = current.astype(np.complex128)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Calculate new values for all single flips
            base_vals = Z @ c
            
            # flip_i_vals = base_vals - 2 * c[i] * Z[:, i]
            # Vectorized:
            # term = 2 * c[:, None] * Z.T (shape n, n_grid)
            # flip_vals = base_vals[None, :] - term (shape n, n_grid)
            
            term = 2 * c[:, None] * Z.T
            flip_vals = base_vals[None, :] - term
            
            # Find sup norm for each flip
            flip_norms = np.max(np.abs(flip_vals), axis=1)
            
            # Find best flip
            best_i = np.argmin(flip_norms)
            best_new_val = flip_norms[best_i]
            
            if best_new_val < current_val - 1e-12:
                # Apply flip
                current[best_i] *= -1
                c[best_i] *= -1
                current_val = best_new_val
            else:
                # No improvement possible with single flips (local minimum)
                break
                
        return current, current_val

    # Multiple random restarts
    np.random.seed(42)
    
    # We can afford more restarts now
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r == 0:
            coeffs = best_coeffs.copy()
        else:
            # Random restart
            coeffs = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n)])
        
        found, val = local_search_vectorized(coeffs, max_iters=5000)
        if val < best_val:
            best_val = val
            best_coeffs = found
    
    # Final refinement
    found, val = local_search_vectorized(best_coeffs, max_iters=10000)
    if val < best_val:
        best_val = val
        best_coeffs = found
        
    return [int(c) for c in best_coeffs]
