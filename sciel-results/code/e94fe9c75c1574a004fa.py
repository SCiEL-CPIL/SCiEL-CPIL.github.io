"""First autocorrelation inequality: minimize normalized max of autoconvolution
using a combination of parametric initialization and local optimization."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
DX = 1.0 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def compute_score(h):
        """Compute the normalized maximum of the autoconvolution."""
        conv = np.convolve(h, h, mode='full')
        integral = np.sum(h) * DX
        norm = integral * integral
        if norm == 0:
            return np.inf
        return np.max(conv) / norm

    # Initial guess: flat function
    h_flat = np.ones(N)
    best_h = h_flat.copy()
    best_score = compute_score(h_flat)
    
    # Generate candidate functions from various families
    candidates = [h_flat]
    
    # Family 1: Cosine bump
    def cosine_bump(center=0.0, width=0.5, amplitude=1.0):
        h = np.zeros(N)
        x = np.linspace(-0.25, 0.25, N+1)[:-1]  # left endpoints
        x_centers = x + DX/2
        for i, xc in enumerate(x_centers):
            d = abs(xc - center)
            if d < width:
                h[i] = amplitude * 0.5 * (1 + np.cos(np.pi * d / width))
            else:
                h[i] = 0
        return h
    
    # Family 2: Rectangular
    def rectangular(center=0.0, half_width=0.125, amplitude=1.0):
        h = np.zeros(N)
        x_centers = np.linspace(-0.25, 0.25, N) + DX/2
        for i, xc in enumerate(x_centers):
            if abs(xc - center) < half_width:
                h[i] = amplitude
            else:
                h[i] = 0
        return h
    
    # Family 3: Gaussian
    def gaussian(center=0.0, sigma=0.05, amplitude=1.0):
        h = np.zeros(N)
        x_centers = np.linspace(-0.25, 0.25, N) + DX/2
        for i, xc in enumerate(x_centers):
            h[i] = amplitude * np.exp(-0.5 * ((xc - center) / sigma) ** 2)
        return h
    
    # Add candidates
    # Cosine bumps with different parameters
    for center in [0.0]:
        for width in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            for amp in [0.5, 1.0, 2.0]:
                candidates.append(cosine_bump(center, width, amp))
    
    # Rectangular
    for half_w in [0.05, 0.1, 0.125, 0.15, 0.2]:
        for amp in [0.5, 1.0, 2.0]:
            candidates.append(rectangular(0.0, half_w, amp))
    
    # Gaussian
    for sigma in [0.02, 0.03, 0.05, 0.07, 0.1]:
        for amp in [0.5, 1.0, 2.0]:
            candidates.append(gaussian(0.0, sigma, amp))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline:
            break
        s = compute_score(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Local refinement using coordinate descent with exact 1D minimization
    # This is more reliable than heuristic steps or gradient-free methods for nonsmooth objectives
    if time.time() < deadline - 5:
        # Multiple passes of coordinate descent
        for pass_num in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for coord in range(N):
                if time.time() > deadline - 5:
                    break
                
                h0_val = best_h[coord]
                
                # Define 1D objective for this coordinate
                def obj_1d(val):
                    h_trial = best_h.copy()
                    h_trial[coord] = max(0.0, val)
                    return compute_score(h_trial)
                
                # Use minimize_scalar to find the best value for this coordinate
                # Bounds: [0, max(h0_val * 2, 1.0)] to allow for significant changes
                bound_max = max(h0_val * 2, 1.0)
                try:
                    res = minimize_scalar(obj_1d, bounds=(0, bound_max), method='bounded',
                                         options={'xatol': 1e-4, 'maxiter': 50})
                    if res.success:
                        new_val = max(0.0, res.x)
                        if abs(new_val - h0_val) > 1e-6:
                            best_h[coord] = new_val
                            new_score = compute_score(best_h)
                            if new_score < best_score:
                                best_score = new_score
                                improved = True
                except Exception:
                    pass
            
            if not improved:
                break
    
    # Final polish: try a few random perturbations to escape local minima
    if time.time() < deadline - 1:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            h_perturbed = best_h.copy()
            # Perturb a random subset of coordinates
            indices = rng.choice(N, size=10, replace=False)
            for idx in indices:
                h_perturbed[idx] = max(0.0, h_perturbed[idx] * (1 + rng.normal(0, 0.1)))
            
            # Do a quick local refinement on the perturbed function
            for coord in range(N):
                if time.time() > deadline - 1:
                    break
                h0_val = h_perturbed[coord]
                
                def obj_1d_pert(val):
                    h_trial = h_perturbed.copy()
                    h_trial[coord] = max(0.0, val)
                    return compute_score(h_trial)
                
                bound_max = max(h0_val * 2, 1.0)
                try:
                    res = minimize_scalar(obj_1d_pert, bounds=(0, bound_max), method='bounded',
                                         options={'xatol': 1e-4, 'maxiter': 20})
                    if res.success:
                        h_perturbed[coord] = max(0.0, res.x)
                except Exception:
                    pass
            
            s_pert = compute_score(h_perturbed)
            if s_pert < best_score:
                best_score = s_pert
                best_h = h_perturbed.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
