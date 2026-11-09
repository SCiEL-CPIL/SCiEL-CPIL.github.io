import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute the evaluation matrix
    # Use a dense grid for local search to ensure we find the global optimum
    M = 10000
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Z[i, k] = exp(i * theta_i * k)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    best_c = None
    best_score = np.inf
    
    def local_search(c_init):
        """
        Performs local search starting from c_init.
        Returns (best_c, best_score).
        """
        c = c_init.astype(float).copy()
        vals = Z @ c
        current_score = np.max(np.abs(vals))
        
        improved = True
        while improved:
            if time.time() > deadline:
                return c, current_score
            improved = False
            for i in range(N):
                # Flip bit i
                old_val = c[i]
                c[i] = -old_val
                # Update vals: vals_new = vals - 2 * c[i] * Z[:, i]
                # Note: c[i] is the NEW value. The change is -2 * NEW * Z[:, i]
                # Wait, P_new = P_old + (c_new - c_old) * z^i = P_old - 2*c_old*z^i = P_old + 2*c_new*z^i
                # Since c_new = -c_old, change is -2*c_old*z^i.
                # Let's stick to: vals_new = vals - 2 * old_val * Z[:, i]
                vals_new = vals - 2 * old_val * Z[:, i]
                new_score = np.max(np.abs(vals_new))
                
                if new_score < current_score:
                    current_score = new_score
                    vals = vals_new
                    improved = True
                else:
                    c[i] = old_val
                    
        return c, current_score

    # List of initializations to try
    inits = []
    
    # 1. GRS polynomial (truncated)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    inits.append(np.array(p[:N], dtype=float))
    
    # 2. Alternating signs
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float))
    
    # 3. All ones
    inits.append(np.ones(N, dtype=float))
    
    # 4. Random restarts
    num_restarts = 500
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        inits.append(np.random.choice([-1, 1], N).astype(float))
        
    for c_init in inits:
        if time.time() > deadline:
            break
            
        c, score = local_search(c_init)
        
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
        # Early stop if we hit the theoretical optimum (approx)
        # Optimum is 1.2583758 * sqrt(24) = 6.154...
        if best_score < 6.1544:
            break

    if best_c is None:
        # Fallback
        best_c = np.ones(N, dtype=int)
    else:
        # Ensure it's an integer array
        best_c = best_c.astype(int)
        
    # Return list of Python integers to avoid numpy type issues
    return [int(x) for x in best_c]
