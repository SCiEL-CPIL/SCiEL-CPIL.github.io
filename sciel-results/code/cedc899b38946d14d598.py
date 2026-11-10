"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n = 8

    def calc_score_from_angles(angles):
        # angles: array of n floats
        # Points on unit circle
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Diameter
        D = np.max(d)
        if D < 1e-10:
            return -np.inf
            
        # Product of ordered pairs
        mask = ~np.eye(n, dtype=bool)
        d_off = d[mask]
        
        # Sum of logs
        log_prod = np.sum(np.log(d_off))
        
        # Scaling factor
        # d'_ij = d_ij * (2/D)
        # prod d'_ij = prod d_ij * (2/D)^{n(n-1)}
        # log(prod d'_ij) = log_prod + n(n-1) * log(2/D)
        
        log_scaled_prod = log_prod + n * (n - 1) * (np.log(2.0) - np.log(D))
        
        return log_scaled_prod

    def neg_log_score(angles):
        # angles shape (n,)
        # We fix angles[0] = 0 to remove rotational symmetry
        # This reduces dimensionality from 8 to 7.
        # However, the evaluator might want distinct points.
        # If angles are very close, points might be identical.
        # We enforce a minimum separation or just rely on the optimizer.
        
        theta_0 = 0.0
        thetas = np.concatenate(([theta_0], angles))
        
        val = calc_score_from_angles(thetas)
        if not np.isfinite(val):
            return 1e10
        return -val

    # Initial guess: Regular octagon
    # Angles: 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # Variables: theta_1 ... theta_7
    init_angles = np.array([np.pi * k / 4 for k in range(1, 8)])
    
    best_val = -neg_log_score(init_angles)
    best_angles = init_angles.copy()

    # Multi-start optimization in Angular Space (1D on circle)
    # This is structurally different from the 2D search in the current program.
    # It exploits the likelihood that the optimal configuration is cyclic.
    
    rng = np.random.default_rng(42)
    
    num_starts = 50
    max_time_per_opt = 2.0
    
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate perturbation
        # Small random noise
        noise = rng.normal(0, 0.1, size=7)
        current_angles = init_angles + noise
        
        # Also try some specific structural deviations
        # e.g., alternating radii? No, we are constrained to circle.
        # The optimal solution for n=8 is known to be the regular octagon?
        # Wait, the prompt says "The paper's constructions show this is possible [to beat 1.0]".
        # For n=8, is the regular octagon optimal?
        # Actually, for n=8, the regular octagon is a local maximum.
        # Is it the global maximum?
        # Literature suggests that for n > 6, the optimal configuration might not be regular.
        # However, for points on a circle, the regular polygon is often the local max.
        # But if we allow points OFF the circle, we might do better.
        
        # Let's reconsider. The "two squares" approach in the current program allows off-circle points.
        # My "circular" assumption might be too restrictive if the global max is off-circle.
        # However, the inspiration says "Restricting the search to the unit circle... Measured: 1 -> 1.001".
        # This implies that even on the circle, one can beat 1.0.
        # How? By deviating from the regular angles?
        # If the points are on a circle, the product of distances is maximized by the regular polygon?
        # Let's check.
        # Product of chords for regular n-gon: constant.
        # If we perturb angles, does the product increase?
        # For n=3, 4, 5, 6, regular is optimal.
        # For n=8?
        # If the regular octagon is a local maximum on the circle, then perturbing angles will decrease the score.
        # So how did the inspiration get 1.001?
        # Maybe the inspiration did NOT restrict to the circle?
        # "Restricting the search to the unit circle... reduces dimensionality... Measured: 1 -> 1.001"
        # This phrasing is ambiguous. It might mean the *program* that got 1.001 used a circle-based search, 
        # or it might mean that the strategy of *allowing* circle-based search (as opposed to 2D) yielded that.
        # Or maybe the optimal configuration for n=8 IS on the circle but NOT regular?
        # Let's test if regular octagon is a max on the circle.
        
        # Let's try to optimize the angles. If it stays at regular, then circle is not enough.
        # But if the inspiration got 1.001, maybe I should try 2D search but with a better optimizer?
        
        # Alternative Strategy: 
        # The current program uses "Two Squares". This is a specific subset of 2D points.
        # A more general 2D search with a better global optimizer (like Differential Evolution or Basin Hopping)
        # might find a configuration that is NOT two squares and NOT on a circle.
        
        # Let's pivot to a general 2D search using Basin Hopping or Differential Evolution, 
        # but with a tighter budget and better initialization.
        # The current program's score is 1.0000. The inspiration got 1.0007.
        
        # Let's try a general 2D optimization with multiple starts and a robust local optimizer.
        
        pass # Skipping the circular optimization loop for now, switching strategy below.

    # NEW STRATEGY: General 2D search with Differential Evolution or Basin Hopping
    # The current program uses Nelder-Mead on a 3D parameter space (2 squares).
    # We will use a 16D parameter space (8 points x 2 coords) with a global optimizer.
    # To make it feasible in 150s, we will use a coarse global search followed by local refinement.
    
    # Objective: Maximize log(product of distances) - n(n-1)*log(diameter)
    # This is scale invariant.
    
    def objective_2d(pts_flat):
        # pts_flat: 16 floats
        pts = pts_flat.reshape((n, 2))
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        mask = ~np.eye(n, dtype=bool)
        d_off = d[mask]
        
        # If any distance is 0, return large penalty
        if np.any(d_off < 1e-6):
            return 1e10
            
        log_prod = np.sum(np.log(d_off))
        val = log_prod - n * (n - 1) * np.log(D)
        
        if not np.isfinite(val):
            return 1e10
        return -val

    # Initial points: Regular octagon scaled to diameter 2
    angles_reg = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    # Scale to diameter 2
    diff_reg = pts_reg[:, np.newaxis, :] - pts_reg[np.newaxis, :, :]
    d_reg = np.sqrt(np.sum(diff_reg * diff_reg, axis=2))
    D_reg = np.max(d_reg)
    pts_reg_scaled = pts_reg * (2.0 / D_reg)
    
    best_pts = pts_reg_scaled
    best_score_val = -objective_2d(pts_reg_scaled.flatten())
    
    # Use Differential Evolution for global search
    # Bounds: [-10, 10] is sufficient since we scale at the end.
    bounds = [(-10, 10)] * 16
    
    # DE can be slow. Let's try Basin Hopping or just multiple Nelder-Mead starts in 2D.
    # Given the 150s limit, DE with 16 dims might be too slow unless we limit generations.
    # Let's try a few random starts with Nelder-Mead in 2D.
    
    rng2 = np.random.default_rng(123)
    
    num_starts_2d = 10
    
    for i in range(num_starts_2d):
        if time.time() > deadline - 20.0:
            break
            
        # Random initialization near the regular octagon
        # Perturb the regular octagon
        noise = rng2.normal(0, 0.5, size=(n, 2))
        init_pts = pts_reg_scaled + noise
        init_flat = init_pts.flatten()
        
        try:
            res = minimize(objective_2d, init_flat, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 1000})
            
            if res.fun < best_score_val:
                best_score_val = res.fun
                best_pts = res.x.reshape((n, 2))
                
        except Exception:
            pass

    # Try a few more specific structural perturbations
    # e.g., "Square + Square" but with optimized parameters (like the current program)
    # But let's also try "Two Rectangles" or just generic 2D.
    
    # Let's do a final polish on the best point found
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective_2d, best_pts.flatten(), method='Powell',
                           options={'maxiter': 1000, 'maxfev': 1000})
            if res.fun < best_score_val:
                best_score_val = res.fun
                best_pts = res.x.reshape((n, 2))
        except Exception:
            pass

    # Finalize: Scale to diameter 2
    diff_final = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    d_final = np.sqrt(np.sum(diff_final * diff_final, axis=2))
    D_final = np.max(d_final)
    
    if D_final < 1e-10:
        # Fallback to regular octagon
        return [[np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)] for k in range(n)]
        
    pts_scaled = best_pts * (2.0 / D_final)
    
    # Check validity
    if not np.all(np.isfinite(pts_scaled)) or np.max(np.abs(pts_scaled)) >= 1e6:
        return [[np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)] for k in range(n)]
        
    return pts_scaled.tolist()

# EVOLVE-BLOCK-END
