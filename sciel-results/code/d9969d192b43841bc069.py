"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _coordinate_descent_polish(h, max_sweeps=100, tol=1e-12):
    """
    Refine h using coordinate descent.
    For each i, optimize h_i to minimize max(overlap_{i-1}, overlap_i, rest_max).
    """
    h = h.copy()
    h = _enforce_sum(h)
    
    for sweep in range(max_sweeps):
        max_change = 0.0
        # Precompute overlaps to know the 'rest_max' for each i efficiently?
        # N is small (100), so recalculating is fine, but we can be smarter.
        # However, changing h_i only affects overlap_{i-1} and overlap_i.
        
        for i in range(N):
            # Get neighbors
            h_prev = h[i-1] if i > 0 else 0.0
            h_next = h[i+1] if i < N-1 else 0.0
            
            # Calculate current overlap values
            # overlap_{i-1} = h_{i-1} * (1 - h_i)
            # overlap_i = h_i * (1 - h_{i+1})
            
            # We need M_others = max(overlap_k for k != i-1, i)
            # To avoid O(N) per i (total O(N^2)), we can just compute full max and 
            # if the max is not at i-1 or i, M_others = global_max.
            # If max is at i-1 or i, we need the second max.
            # Given N=100, O(N^2) = 10,000 ops per sweep is trivial.
            
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            
            # Find max excluding i-1 and i
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            if i > 0: mask[i-1] = False
            if i < N-1: mask[i+1] = False # Note: overlap_i depends on h_i and h_{i+1}. 
            # Wait, overlap index k is h_k (1-h_{k+1}).
            # Changing h_i affects:
            # overlap_{i-1} = h_{i-1}(1-h_i)
            # overlap_i = h_i(1-h_{i+1})
            # So we exclude indices i-1 and i from the "others".
            
            if i == 0:
                mask[0] = False # overlap_0
                # overlap_{-1} doesn't exist
            elif i == N-1:
                mask[N-1] = False # overlap_{N-1} = h_{N-1}(1-0) = h_{N-1}
                mask[N-2] = False # overlap_{N-2} = h_{N-2}(1-h_{N-1})
            else:
                mask[i] = False
                mask[i-1] = False
            
            M_others = np.max(overlap[mask]) if np.any(mask) else 0.0
            
            A = h_prev
            B = h_next
            
            # Minimize max( A(1-x), x(1-B), M_others )
            # Let f(x) = max(A - Ax, x - Bx, M_others)
            # Intersection of A-Ax and x-Bx:
            # A - Ax = x - Bx  => A = x(1 - B + A) => x_cross = A / (1 + A - B)
            
            denom = 1.0 + A - B
            if abs(denom) < 1e-12:
                x_new = h[i]
            else:
                x_cross = A / denom
                
                # Clamp to [0, 1]
                if x_cross < 0.0:
                    x_candidate = 0.0
                elif x_cross > 1.0:
                    x_candidate = 1.0
                else:
                    x_candidate = x_cross
                
                # Evaluate f at x_candidate
                val1 = A * (1.0 - x_candidate)
                val2 = x_candidate * (1.0 - B)
                f_val = max(val1, val2, M_others)
                
                # Check if we can do better by pushing towards M_others?
                # If f_val > M_others, the bottleneck is the lines. x_cross minimizes the max of the lines.
                # If f_val == M_others, it means both lines are <= M_others at x_cross.
                # In that case, any x in the range where both lines <= M_others is optimal for this step.
                # We stick with x_cross as it's the "center" of the flat bottom.
                
                # However, if x_cross is outside [0,1], we check boundaries.
                # If x_cross < 0, max is at x=0?
                # f(0) = max(A, 0, M_others).
                # f(1) = max(0, 1-B, M_others).
                
                # Let's just evaluate f at x_cross, 0, 1 and pick the best.
                # This is robust.
                candidates = [x_candidate, 0.0, 1.0]
                best_x = x_candidate
                best_val = float('inf')
                
                for xc in candidates:
                    v1 = A * (1.0 - xc)
                    v2 = xc * (1.0 - B)
                    v = max(v1, v2, M_others)
                    if v < best_val:
                        best_val = v
                        best_x = xc
                
                x_new = best_x
            
            # Update h[i]
            change = abs(x_new - h[i])
            max_change = max(max_change, change)
            h[i] = x_new
            
            # Enforce sum constraint locally? 
            # Coordinate descent on constrained problems usually requires re-projection.
            # But changing h_i changes the sum.
            # We can adjust h[i] and then spread the error? 
            # Better: Just enforce sum at the end of the sweep.
            
        # Enforce sum and bounds after each sweep
        h = _enforce_sum(h)
        
        if max_change < tol:
            break
            
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    current_score = float('inf')
    h_final = np.zeros(N)
    
    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Initial guesses
    guesses = []
    
    # 1. 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    guesses.append(h0)
    
    # 2. Sinusoidal guesses
    for freq in [1, 2, 3, 5]:
        for phase in [0, np.pi/2, np.pi]:
            h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            guesses.append(h_sin)
            
    # 3. Random smooth guesses
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.rand(N)
        # Smooth it
        kernel = np.ones(5) / 5
        h_rand = np.convolve(h_rand, kernel, mode='same')
        h_rand = 0.2 + 0.6 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
        guesses.append(h_rand)

    # Optimization Loop
    for idx, h_guess in enumerate(guesses):
        if time.time() > deadline - 10.0:
            break
            
        h_guess = _enforce_sum(h_guess)
        
        # Step 1: SLSQP with smooth objective
        try:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # Use high alpha for smoothness
                smax = _smooth_max(overlap, alpha=200.0)
                return t + 5000.0 * np.maximum(0.0, smax - t)**2
            
            x0 = np.concatenate([h_guess, [_score(h_guess) + 0.1]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0
            
            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)
            
            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]
            
            result = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            
            if result.fun < 100.0:
                h_opt = result.x[:N].copy()
                _try_update(h_opt)
                
        except Exception:
            pass
        
        # Step 2: Coordinate Descent Polish on the best result so far (or current guess if SLSQP failed)
        # We polish the best result found in this iteration, or the guess if SLSQP didn't improve
        h_to_polish = h_final.copy() if current_score < _score(h_guess) else h_guess.copy()
        
        try:
            if time.time() < deadline - 5.0:
                h_polished = _coordinate_descent_polish(h_to_polish, max_sweeps=50)
                _try_update(h_polished)
        except Exception:
            pass

    # Final Polish on the global best
    try:
        if time.time() < deadline - 2.0:
            h_final = _coordinate_descent_polish(h_final, max_sweeps=100)
            _try_update(h_final)
    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
