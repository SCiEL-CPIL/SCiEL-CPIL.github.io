"""Root finder for 3 fixed 2D systems using Globally Convergent Newton (GCR) with Adaptive Hessian."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 490
    
    def safe_eval(x):
        nonlocal call_counter
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, float('inf')
            call_counter[0] += 1
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, float('inf')

    # Strategy: Globally Convergent Newton (GCR)
    # 1. Start with a robust initial Hessian approximation (Identity or diagonal scaling)
    # 2. Use a line search that guarantees global convergence (Wolfe conditions or simple sufficient decrease)
    # 3. Update Hessian using Broyden's Good Method (rank-1 update)
    # 4. If the method stalls or diverges, restart from a different point with a fresh Hessian
    
    best_x = x0_arr
    best_norm = float('inf')
    
    # Generate a set of starting points
    # 1. x0
    # 2. Origin
    # 3. Points on a circle around x0 with increasing radii
    # 4. Random points
    
    starts = [x0_arr.copy(), np.zeros(2)]
    
    # Add points on circles
    radii = [0.5, 1.0, 2.0, 5.0, 10.0]
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for r in radii:
        for a in angles:
            starts.append(x0_arr + r * np.array([np.cos(a), np.sin(a)]))
    
    # Add some random points
    rng = np.random.RandomState(42)
    for _ in range(10):
        starts.append(rng.uniform(-10, 10, 2))
    
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
            
        # Initialize Hessian as Identity
        H = np.eye(2)
        x = x_start.copy()
        
        Fx, norm_F = safe_eval(x)
        if Fx is None:
            continue
            
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x.copy()
            
        if norm_F <= 1e-8:
            break
            
        converged = False
        
        for _ in range(50):  # Max iterations per restart
            if time.time() - start_time > time_limit:
                break
            if call_counter[0] >= max_calls:
                break
                
            # Newton step: solve H * s = -Fx
            try:
                s = np.linalg.solve(H, -Fx)
            except np.linalg.LinAlgError:
                # If singular, use least squares
                try:
                    s = np.linalg.lstsq(H, -Fx, rcond=None)[0]
                except:
                    s = -Fx  # Gradient descent fallback
                    
            if not np.all(np.isfinite(s)):
                break
                
            # Line search: find alpha in (0, 1] such that ||F(x + alpha*s)|| < (1 - c*alpha) * ||F(x)||
            alpha = 1.0
            c = 1e-4
            x_new = None
            F_new = None
            norm_new = float('inf')
            
            for _ in range(20):
                x_trial = x + alpha * s
                F_trial, norm_trial = safe_eval(x_trial)
                
                if F_trial is None:
                    alpha *= 0.5
                    continue
                    
                if norm_trial < (1 - c * alpha) * norm_F:
                    x_new = x_trial
                    F_new = F_trial
                    norm_new = norm_trial
                    break
                else:
                    alpha *= 0.5
                    
            if x_new is None:
                # No improvement found, break
                break
                
            # Update Hessian (Broyden's Good Method)
            delta_x = x_new - x
            delta_F = F_new - Fx
            denom = np.dot(delta_x, delta_x)
            
            if denom > 1e-12:
                H = H + np.outer(delta_F - H @ delta_x, delta_x) / denom
            
            x = x_new
            Fx = F_new
            norm_F = norm_new
            
            if norm_F < best_norm:
                best_norm = norm_F
                best_x = x.copy()
                
            if norm_F <= 1e-8:
                converged = True
                break
                
        if converged:
            break
            
    return best_x
# EVOLVE-BLOCK-END
