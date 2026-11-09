"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def design(problem):
    """
    Design a broadband antireflection coating using global optimization (Differential Evolution)
    followed by local polishing (L-BFGS-B).
    """
    t_start = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    DEADLINE = 140.0
    
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    n_L = materials["L"]
    n_H = materials["H"]
    band = problem.get("band_nm", [400, 800])
    lam_min, lam_max = band[0], band[1]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    # Wavelength grid for evaluation
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    
    def compute_reflectance(labels, thicknesses_nm):
        """
        Exact characteristic-matrix method for normal incidence.
        Returns mean reflectance in percent.
        """
        n_layers = len(labels)
        if n_layers == 0:
            # Bare substrate
            r = (ambient - substrate) / (ambient + substrate)
            return float(np.abs(r)**2 * 100.0)
        
        # Pre-compute indices for this stack
        n_stack = np.array([materials[lab] for lab in labels])
        
        # M is 2x2 matrix per wavelength. We accumulate M_total = M1 @ M2 ... Mn
        # M_j = [[cos(delta), 1j/(n_j*sin(delta))], [1j*n_j*sin(delta), cos(delta)]]
        
        # Initialize identity
        M = np.eye(2, dtype=complex) # This is for a single wavelength, but we vectorize over wavelengths
        
        # We need to handle the matrix multiplication vectorized over wavelengths.
        # M is 2x2x(n_wl)
        M00 = np.ones(n_wl, dtype=complex)
        M01 = np.zeros(n_wl, dtype=complex)
        M10 = np.zeros(n_wl, dtype=complex)
        M11 = np.ones(n_wl, dtype=complex)
        
        for i in range(n_layers):
            n_j = n_stack[i]
            d = thicknesses_nm[i]
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            # Avoid division by zero if s is very small (shouldn't happen for d>0, but d can be 0)
            # If d=0, delta=0, s=0. The matrix becomes singular? 
            # Actually if d=0, the layer is absent. 
            # M_j = [[1, 0], [0, 1]] if we take limits?
            # cos(0)=1, sin(0)=0. 1/sin(0) is inf. 
            # We should skip layers with d=0 or handle them.
            # For optimization, d=0 is a valid boundary.
            # If s is close to 0, we can approximate. But usually d is not exactly 0 in optimal solutions.
            # Let's add a small epsilon to s to prevent inf, or skip if d < 1e-6.
            
            if d < 1e-6:
                continue # Layer is effectively absent
            
            M01_j = 1j / (n_j * s)
            M10_j = 1j * n_j * s
            M11_j = c
            M00_j = c
            
            # M_new = M @ M_j
            # M is represented by M00, M01, M10, M11
            new00 = M00 * M00_j + M01 * M10_j
            new01 = M00 * M01_j + M01 * M11_j
            new10 = M10 * M00_j + M11 * M10_j
            new11 = M10 * M01_j + M11 * M11_j
            
            M00, M01, M10, M11 = new00, new01, new10, new11
            
        # Admittance Y = (M11 + M12 * Y_sub) / (M21 + M22 * Y_sub)
        # Note: My matrix convention is M = [[M00, M01], [M10, M11]]
        # Standard CM: M = [[M11, M12], [M21, M22]]
        # So M11 (standard) = M00 (my), M12 (standard) = M01 (my), etc.
        # Y = (M00 + M01 * Y_sub) / (M10 + M11 * Y_sub)
        
        Y_sub = substrate
        Y = (M00 + M01 * Y_sub) / (M10 + M11 * Y_sub)
        
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)

    def objective(x):
        """
        Objective function for optimization.
        x: array of thicknesses.
        """
        # Enforce total thickness constraint by scaling
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
        
        # Clip to max_layer (DE handles bounds, but scaling might push out? No, scaling reduces.
        # But if one element was max and others 0, scaling doesn't change it.
        # If one element is > max_layer, we need to clip.
        # However, if we clip, we might violate sum <= max_total.
        # Let's just rely on DE bounds for individual layers and scaling for sum.
        # If scaling causes an element to exceed max_layer? No, scaling reduces values.
        # If an element was < max_layer, it stays < max_layer.
        
        # Ensure non-negative
        x = np.clip(x, 0.0, max_layer)
        
        # If scaling made sum > max_total again? No, scaling is proportional.
        # But if we clip after scaling, sum might be < max_total, which is fine.
        
        return compute_reflectance(labels, list(x))

    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Candidate layer counts
    # We want symmetric stacks.
    # For even N, half is N/2. Full = half + reverse(half).
    # For odd N, it's trickier. Let's stick to even numbers for simplicity and symmetry.
    # Max layers is 20.
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Sort by complexity? Or just iterate?
    # Smaller stacks are faster to evaluate and optimize.
    # Let's try smaller ones first to get a baseline, then larger ones if time permits.
    
    for n_layers in layer_counts:
        if time.time() > DEADLINE - 10:
            break
            
        if n_layers > max_layers:
            continue
            
        # Construct symmetric labels
        half_n = n_layers // 2
        # Pattern: L H L H ...
        # Start with L on air side.
        half_labels = []
        for i in range(half_n):
            half_labels.append("L" if i % 2 == 0 else "H")
        full_labels = half_labels + list(reversed(half_labels))
        labels = full_labels
        
        n_var = len(labels)
        
        # Bounds for DE
        bounds = [(0.0, max_layer)] * n_var
        
        # Initial population strategy for DE?
        # DE generates random points in bounds.
        # We can provide a seed or use the default.
        
        # Time allocation for this stack
        time_remaining = DEADLINE - time.time()
        if time_remaining < 5:
            continue
            
        # Adjust maxiter and popsize based on time and dimension
        # Rough estimate: cost per eval is ~ n_layers * n_wl ops.
        # Let's cap popsize to 20 for speed, maxiter to 100.
        popsize = 20
        maxiter = 100
        
        try:
            # Differential Evolution
            # Use 'best1bin' mutation
            # Tolerance for convergence
            res_de = differential_evolution(
                objective, 
                bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                tol=1e-8,
                seed=42,
                polish=False # We will polish manually
            )
            
            if time.time() > DEADLINE - 5:
                # Too slow, use the best found so far if any
                if best_score == float('inf') and res_de.fun < float('inf'):
                    best_score = res_de.fun
                    best_labels = labels
                    best_thick = list(res_de.x)
                continue
                
            x_best = res_de.x
            
            # Local Polish with L-BFGS-B
            # L-BFGS-B handles bounds well.
            # We need to handle the sum constraint.
            # We can define a wrapped objective that projects x to feasible set.
            
            def obj_polish(x):
                s = np.sum(x)
                if s > max_total:
                    x = x * (max_total / s)
                x = np.clip(x, 0.0, max_layer)
                return compute_reflectance(labels, list(x))
                
            try:
                res_lm = minimize(
                    obj_polish, 
                    x_best, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_lm.fun < res_de.fun:
                    x_best = res_lm.x
                    score_val = res_lm.fun
                else:
                    score_val = res_de.fun
            except Exception:
                score_val = res_de.fun
                
            # Final validation and clipping
            x_final = np.clip(x_best, 0.0, max_layer)
            s_final = np.sum(x_final)
            if s_final > max_total:
                x_final = x_final * (max_total / s_final)
                x_final = np.clip(x_final, 0.0, max_layer)
                
            # Recompute score to be sure
            final_score = compute_reflectance(labels, list(x_final))
            
            if final_score < best_score:
                best_score = final_score
                best_labels = labels
                best_thick = list(x_final)
                
        except Exception as e:
            pass

    # Fallback if no valid result found
    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
        
    return best_labels, best_thick
# EVOLVE-BLOCK-END
