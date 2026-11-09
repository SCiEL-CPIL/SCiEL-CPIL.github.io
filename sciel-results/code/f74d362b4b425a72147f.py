"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Strategy: Trust-region Damped Newton with Random Restarts
    # 1. Use a strict time budget (145s) to avoid timeout.
    # 2. Start with x0.
    # 3. If Newton fails (singular, non-descent, or no convergence), 
    #    perform random restarts around x0 or previous best points.
    # 4. Each Newton iteration uses adaptive damping (line search) to ensure 
    #    the norm decreases.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False
    
    def newton_damped(F, x0, max_iter=50, tol=1e-12):
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
            
            try:
                Fx = F(x)
            except Exception:
                break
            
            if not np.all(np.isfinite(Fx)):
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            if current_norm <= tol:
                return x
            
            # Compute Jacobian via finite differences
            # Use complex-step if possible? No, F might not support complex.
            # Use central differences for better accuracy? 
            # Forward difference is cheaper (3 calls vs 5). 
            # Let's stick to forward difference but with adaptive eps.
            
            eps = 1e-6
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                try:
                    J[:, j] = (F(x + e) - Fx) / eps
                except Exception:
                    break
            
            if not np.all(np.isfinite(J)):
                break
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Damped Newton Step (Line Search)
            # Try lambda = 1.0, 0.5, 0.25, ...
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue
                
                if not np.all(np.isfinite(F_new)):
                    continue
                
                new_norm = norm(F_new)
                # Armijo condition: new_norm < current_norm * (1 - c * lam * norm(dx))
                # Simplified: just require decrease
                if new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break
            
            if not step_found:
                # If no step reduces the norm, we are stuck or at a singularity.
                # Break to allow restart.
                break
                
            # Convergence check on step size
            if norm(dx) < 1e-10:
                break
        
        return best_x

    # Phase 1: Try Newton from x0
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    x_best = newton_damped(F, x0)
    
    if is_good(x_best):
        return x_best
    
    # Phase 2: Random Restarts
    # Generate random points in a sphere around x0
    # Scale the radius based on x0 magnitude or default to 1.0
    scale = max(1.0, norm(x0))
    
    # Try a few deterministic offsets first if random is not allowed? 
    # "Random" is allowed. 
    # We will try a set of candidate points.
    
    candidates = []
    
    # 1. Zero
    candidates.append(np.array([0.0, 0.0]))
    
    # 2. x0 scaled
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    
    # 3. Random points
    # Use a fixed seed for reproducibility in case the evaluator re-runs?
    # The task says "hidden systems", implying we don't know them.
    # Randomness is fine as long as we are fast.
    rng = np.random.default_rng(42)
    
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0]:
        for _ in range(5):
            # Random direction
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)
            
    # Sort candidates by distance to x0? Or just iterate.
    # Iterate until time runs out or success.
    
    for c in candidates:
        if time_left() < 1.0:
            break
            
        if not np.all(np.isfinite(c)):
            continue
            
        x_try = newton_damped(F, c)
        
        if is_good(x_try):
            return x_try
            
        # Keep track of the best norm found so far
        try:
            F_try = F(x_try)
            if np.all(np.isfinite(F_try)):
                n_try = norm(F_try)
                if n_try < norm(F(x_best)):
                    x_best = x_try
        except Exception:
            pass

    return x_best
# EVOLVE-BLOCK-END
