"""Improved broadband AR coating design using multi-start local optimization 
on diverse stack topologies."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    using multi-start local optimization on diverse stack topologies."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 10.0  # 10s safety margin
    
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
    
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    
    def compute_reflectance(labels, thicknesses_nm):
        """Exact characteristic-matrix method for normal incidence, lossless non-dispersive.
        Returns mean reflectance in percent over the band."""
        n_layers = len(labels)
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d < 1e-9:
                continue
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            i_c = 1j
            M00 = c
            M01 = i_c / (n_j * s)
            M10 = i_c * n_j * s
            M11 = c
            new00 = M[0,0]*M00 + M[0,1]*M10
            new01 = M[0,0]*M01 + M[0,1]*M11
            new10 = M[1,0]*M00 + M[1,1]*M10
            new11 = M[1,0]*M01 + M[1,1]*M11
            M = np.array([[new00, new01],[new10, new11]])
        Y_sub = substrate
        Y = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)
    
    def project(x):
        """Project x onto feasible region: 0 <= x_i <= max_layer, sum <= max_total."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return x
    
    def objective(x, labels):
        x = project(x)
        return compute_reflectance(labels, list(x))
    
    def optimize_from_start(labels, x0, max_iter=200):
        """Run L-BFGS-B from a specific starting point."""
        from scipy.optimize import minimize
        
        x0 = project(x0)
        best_x = x0.copy()
        best_val = objective(best_x, labels)
        
        bounds = [(0.0, max_layer)] * len(labels)
        
        try:
            res = minimize(lambda x: objective(x, labels), x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_x = project(res.x)
                best_val = objective(best_x, labels)
        except Exception:
            pass
        
        return best_x, best_val
    
    def generate_candidate_stacks():
        """Generate a list of (labels, initial_thicknesses) candidates."""
        candidates = []
        center = 0.5 * (lam_min + lam_max)
        
        # Candidate 1: Symmetric alternating L H L H ...
        # Try a few layer counts
        for n_layers in [10, 12, 14]:
            if n_layers > max_layers:
                continue
            half = n_layers // 2
            half_labels = []
            for i in range(half):
                half_labels.append("L" if i % 2 == 0 else "H")
            labels = half_labels + list(reversed(half_labels))
            
            x0 = np.zeros(len(labels))
            for i, lab in enumerate(labels):
                n_i = materials[lab]
                x0[i] = center / (4.0 * n_i)
            x0 = project(x0)
            candidates.append((labels, x0))
            
        # Candidate 2: Thick-Thin pairs (Wood's coating style)
        # Pattern: H L L H L L H ... or similar
        # Let's try a sequence of H-L pairs
        for n_pairs in [4, 5, 6]:
            n_layers = n_pairs * 2
            if n_layers > max_layers:
                continue
            labels = []
            for i in range(n_pairs):
                labels.append("H")
                labels.append("L")
            # Ensure it ends with L to be closer to substrate? 
            # Actually, standard Wood's is often H L L H L L ...
            # Let's try starting with H and ending with L
            # The above generates H L H L H L ...
            # Let's try a specific pattern known for broadband: 
            # H L L H L L H (7 layers) -> symmetric? No.
            # Let's just use the H L H L pattern but optimize it.
            
            x0 = np.zeros(len(labels))
            # Initial guess: H layers thicker, L layers thinner?
            # Or just quarter wave
            for i, lab in enumerate(labels):
                n_i = materials[lab]
                x0[i] = center / (4.0 * n_i)
            x0 = project(x0)
            candidates.append((labels, x0))
            
        # Candidate 3: Geometric progression thicknesses
        # Labels: alternating L H L H ...
        for n_layers in [12, 16]:
            if n_layers > max_layers:
                continue
            labels = []
            for i in range(n_layers):
                labels.append("L" if i % 2 == 0 else "H")
            
            x0 = np.zeros(len(labels))
            # Geometric progression: d_i = d_0 * r^i
            # Choose r such that total thickness is ~ max_total
            # Sum = d_0 * (1 - r^n) / (1 - r) = max_total
            # Let's try r = 1.1 or 1.2
            for r in [1.1, 1.2]:
                x0 = np.zeros(n_layers)
                # Normalize to fit max_total
                # d_i = d_0 * r^i
                # Sum = d_0 * sum(r^i)
                sum_r = np.sum([r**i for i in range(n_layers)])
                d_0 = max_total / sum_r
                for i in range(n_layers):
                    x0[i] = d_0 * (r ** i)
                x0 = project(x0)
                candidates.append((labels, x0))
                
        return candidates
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    candidates = generate_candidate_stacks()
    
    # Shuffle candidates to ensure diverse starts are tried early
    random.shuffle(candidates)
    
    for labels, x0 in candidates:
        if time.time() > DEADLINE:
            break
        
        # Run optimization
        opt_x, opt_val = optimize_from_start(labels, x0, max_iter=150)
        
        # Validate
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in opt_x):
            continue
        if sum(opt_x) > max_total + 1e-6:
            opt_x = opt_x * (max_total / sum(opt_x))
        
        val = compute_reflectance(labels, list(opt_x))
        if val < best_score:
            best_score = val
            best_labels = labels
            best_thick = list(opt_x)
            
    # If we have time, try a final polish on the best result with Nelder-Mead
    if best_labels is not None and time.time() < DEADLINE - 20:
        try:
            from scipy.optimize import minimize
            labels = best_labels
            x0 = np.array(best_thick)
            bounds = [(0.0, max_layer)] * len(labels)
            res = minimize(lambda x: objective(x, labels), x0, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
            if res.fun < best_score:
                best_x = project(res.x)
                best_val = compute_reflectance(labels, list(best_x))
                if best_val < best_score:
                    best_score = best_val
                    best_thick = list(best_x)
        except Exception:
            pass
            
    # Fallback
    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
