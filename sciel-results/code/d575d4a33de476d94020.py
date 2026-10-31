"""
Erdos minimum overlap optimization.
Uses FFT-based autocorrelation for fast overlap evaluation.
Employs Differential Evolution for global search and SLSQP for local refinement.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100
DEADLINE = 120.0
SAFETY_MARGIN = 10.0  # seconds

def _compute_overlaps_fft(h):
    """
    Compute the overlap C(k) for all shifts k using FFT.
    C(k) = sum_i h[i] * (1 - h[(i+k) % N])
         = sum_i h[i] - sum_i h[i] h[(i+k) % N]
         = N/2 - autocorr(k)
    autocorr(k) = sum_i h[i] h[(i+k) % N]
    """
    # Autocorrelation via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # Due to floating point errors, autocorr might have tiny imaginary parts or negatives
    autocorr = np.real(autocorr)
    overlaps = N / 2.0 - autocorr
    return overlaps

def _smooth_max(x, t=10.0):
    """Smooth approximation of the maximum."""
    x_max = np.max(x)
    if x_max == -np.inf:
        return 0.0
    return x_max + np.log(np.sum(np.exp(t * (x - x_max)))) / t

def _objective(h):
    """
    Objective function to minimize.
    Minimizes the smooth max of the overlaps.
    """
    # Ensure h is within bounds and sums to N/2 during optimization?
    # SLSQP handles constraints, but for DE we might need to project.
    # For SLSQP, we pass raw h and use constraints.
    # However, if h goes out of [0,1], overlaps can be weird.
    # It is safer to clip in the objective if using DE, but SLSQP bounds handle it.
    
    # For SLSQP, h is already constrained by bounds and equality constraint.
    # But to be safe against numerical drift, we can clip.
    h_clipped = np.clip(h, 0.0, 1.0)
    
    overlaps = _compute_overlaps_fft(h_clipped)
    return _smooth_max(overlaps, t=20.0)

def _raw_score(h):
    """
    Compute the true maximum overlap for final scoring.
    """
    h_clipped = np.clip(h, 0.0, 1.0)
    overlaps = _compute_overlaps_fft(h_clipped)
    return np.max(overlaps)

def _enforce_sum(h, target_sum=N/2.0):
    """
    Adjust h to have the exact target sum, maintaining [0,1] bounds if possible.
    """
    s = np.sum(h)
    if s == 0:
        return h
    # Simple scaling
    h_scaled = h * (target_sum / s)
    # Clip
    h_clipped = np.clip(h_scaled, 0.0, 1.0)
    # If clipping changed the sum, do iterative adjustment
    for _ in range(50):
        s = np.sum(h_clipped)
        diff = target_sum - s
        if abs(diff) < 1e-9:
            break
        # Distribute diff evenly
        h_clipped += diff / N
        h_clipped = np.clip(h_clipped, 0.0, 1.0)
    return h_clipped

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant 0.5
    h_const = np.full(N, 0.5)
    candidates.append(h_const)
    
    # 2. Block structure (Inspiration 1)
    A = (3.0 - np.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h_block = np.zeros(N)
    h_block[:N//2] = A
    h_block[N//2:] = B
    h_block = _enforce_sum(h_block)
    candidates.append(h_block)
    
    # 3. Alternating 0 and 1 (approx)
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt[1::2] = 0.0
    # Sum is 50, so it's already valid.
    candidates.append(h_alt)
    
    # 4. Random noise
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand = _enforce_sum(h_rand)
    candidates.append(h_rand)
    
    best_h = None
    best_score = np.inf
    
    # Phase 1: Local Refinement with SLSQP from each candidate
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    bounds = [(0.0, 1.0)] * N
    
    for h0 in candidates:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(
                _objective,
                h0.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            if res.success or (hasattr(res, 'fun') and res.fun < best_score):
                h_opt = res.x
                h_opt = _enforce_sum(h_opt)
                score = _raw_score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception as e:
            # If SLSQP fails, use the initial candidate if it's valid
            pass

    # If we haven't found a good solution or want to explore more, use DE
    # DE is expensive, so we use it only if time permits and we have a decent starting point
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 20:
        # Use the best found so far as a center for DE? Or just random?
        # DE with small population on 100D
        # We will use a custom objective for DE that projects to valid space
        def de_objective(h):
            h_proj = _enforce_sum(h)
            return _objective(h_proj)
            
        # DE bounds
        de_bounds = [(0.0, 1.0)] * N
        
        try:
            # Small population to fit in time
            # maxiter 20, popsize 10 -> 200 evaluations. Each eval is ~1ms. Total < 1s?
            # Actually FFT is fast. Let's allow more.
            res_de = differential_evolution(
                de_objective,
                de_bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                polish=False, # We will polish manually
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < best_score:
                h_de = res_de.x
                h_de = _enforce_sum(h_de)
                
                # Polish with SLSQP
                try:
                    res_polish = minimize(
                        _objective,
                        h_de.copy(),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
                    )
                    if res_polish.success or res_polish.fun < best_score:
                        h_opt = res_polish.x
                        h_opt = _enforce_sum(h_opt)
                        score = _raw_score(h_opt)
                        if score < best_score:
                            best_score = score
                            best_h = h_opt.copy()
                except Exception:
                    pass
        except Exception:
            pass

    # Final Polish with more iterations if time remains
    if best_h is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        try:
            res_final = minimize(
                _objective,
                best_h.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            if res_final.success or res_final.fun < best_score:
                h_opt = res_final.x
                h_opt = _enforce_sum(h_opt)
                score = _raw_score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure final result is valid
    if best_h is None:
        best_h = np.full(N, 0.5)
    else:
        best_h = _enforce_sum(best_h)
        
    # Final verification
    # The evaluator checks sum == len/2
    # Our _enforce_sum should guarantee this to high precision
    
    return best_h
