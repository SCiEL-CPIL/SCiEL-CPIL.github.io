import numpy as np

def construct():
    N = 24
    # Use a dense grid for fast approximate evaluation
    M = 4096
    angles = 2 * np.pi * np.arange(M) / M
    k_idx = np.arange(N)
    Z = np.exp(1j * np.outer(angles, k_idx))
    
    # Function to compute precise max norm with refinement
    def precise_max_norm(c):
        # c is a numpy array of shape (24,)
        # First, find approximate peaks on the grid
        P = Z @ c
        abs_P = np.abs(P)
        
        # Get top 10 indices
        top_indices = np.argsort(abs_P)[-10:][::-1]
        
        refined_max = 0.0
        for idx in top_indices:
            # Golden section search for local max of |P(e^{i theta})|
            delta = 2 * np.pi / M
            a = angles[idx] - delta
            b = angles[idx] + delta
            
            # Ensure we don't cross the branch cut in a way that breaks continuity?
            # For local search, it's fine.
            
            def f(theta):
                z = np.exp(1j * theta)
                return np.abs(np.dot(c, z ** k_idx))
            
            gr = (np.sqrt(5) - 1) / 2
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(40):
                if f1 < f2:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    f2 = f(x2)
                else:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - gr * (b - a)
                    f1 = f(x1)
                    
            peak_val = max(f1, f2)
            if peak_val > refined_max:
                refined_max = peak_val
                
        return refined_max

    # Function to compute fast approximate max norm
    def fast_max_norm(c):
        P = Z @ c
        return np.max(np.abs(P))

    best_score = np.inf
    best_coeffs = None
    
    # Number of restarts. We need to be careful with time.
    # Each hill climb step is O(N^2 * M) roughly due to vectorization?
    # Actually, calculating all 24 flips:
    # V is (M, N). Z is (M, N).
    # V_k = V - 2 * c_k * Z[:, k:k+1]
    # This is (M, N) - (M, 1) broadcast to (M, N).
    # Then max over axis 0 for each of the 24 columns.
    # This is very fast in NumPy.
    
    # Let's run 200 restarts.
    np.random.seed(42) # For reproducibility if needed, but random is better for search
    num_restarts = 200
    
    # Initial guesses: Random and Golay-Rudin-Shapiro
    # Generate GR-S
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = np.array(p[:N], dtype=float)
    
    for r in range(num_restarts):
        if r == 0:
            current_coeffs = grs_coeffs.copy()
        else:
            # Random initialization
            current_coeffs = np.random.choice([-1, 1], size=N).astype(float)
        
        # Hill Climbing
        improved = True
        # Limit steps to avoid infinite loops (though it should converge quickly)
        max_steps = 500
        step = 0
        while improved and step < max_steps:
            improved = False
            step += 1
            
            # Calculate current values
            V = Z @ current_coeffs
            abs_V = np.abs(V)
            current_max = np.max(abs_V)
            
            # Evaluate all 24 possible flips
            # We want to find k that minimizes max(|V - 2*c_k*Z[:,k]|)
            # Let's vectorize this.
            # c_k is a scalar. Z[:, k] is a vector.
            # We can create a matrix of all flips?
            # flips[k] = V - 2 * current_coeffs[k] * Z[:, k]
            # Shape: (N, M)
            
            # Z.T is (N, M).
            # current_coeffs is (N,).
            # We need to broadcast.
            # term = 2 * current_coeffs[:, None] * Z.T
            # new_vals = V[None, :, :] - term[:, None, :] # This creates (N, M)
            
            # Wait, V is (M,). We need to compare each row of (N, M) to find max over M.
            # Let's do it column by column for clarity, or fully vectorized.
            # Fully vectorized:
            # term = 2 * current_coeffs[:, np.newaxis] * Z.T # Shape (N, M)
            # candidate_vals = V[np.newaxis, :] - term # Shape (N, M)
            # candidate_maxes = np.max(np.abs(candidate_vals), axis=1) # Shape (N,)
            
            term = 2 * current_coeffs[:, np.newaxis] * Z.T
            candidate_vals = V[np.newaxis, :] - term
            candidate_maxes = np.max(np.abs(candidate_vals), axis=1)
            
            # Find the best flip
            best_k = np.argmin(candidate_maxes)
            best_new_max = candidate_maxes[best_k]
            
            if best_new_max < current_max - 1e-12:
                # Apply flip
                current_coeffs[best_k] *= -1
                improved = True
        
        # After convergence, refine the best solution
        score = precise_max_norm(current_coeffs)
        
        if score < best_score:
            best_score = score
            best_coeffs = current_coeffs.copy()
            
        # Early stopping if we hit the known optimum?
        # Optimum is ~1.2583758
        if best_score < 1.2583759:
            break

    return best_coeffs.astype(int).tolist()
