# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    directly against the exact characteristic-matrix reflectance objective."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 5.0  # leave 5s safety margin
    
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
        if n_layers == 0:
            r0 = (ambient - substrate) / (ambient + substrate)
            return float(np.abs(r0)**2 * 100.0)
            
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
    
    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence. For even n, half-stack mirrored."""
        if n_layers % 2 != 0:
            n_layers -= 1
        if n_layers <= 0:
            return []
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def objective_wrapper(labels, x):
        """Objective function with constraint projection."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return compute_reflectance(labels, list(x))
    
    def optimize_stack(n_layers, n_restarts=4):
        """Optimize a stack of given size using SA then L-BFGS-B then Nelder-Mead."""
        labels = make_symmetric_labels(n_layers)
        if len(labels) == 0:
            return None
        n_var = len(labels)
        
        center = 0.5*(lam_min + lam_max)
        
        def objective(x):
            return objective_wrapper(labels, x)
        
        best_x = None
        best_val = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            # Initialization
            if restart == 0:
                x0 = np.zeros(n_var)
                for i, lab in enumerate(labels):
                    n_i = materials[lab]
                    x0[i] = center / (4.0 * n_i)
            elif restart == 1:
                # Try a "thicker" initial guess
                x0 = np.zeros(n_var)
                for i, lab in enumerate(labels):
                    n_i = materials[lab]
                    x0[i] = center / (2.0 * n_i)
            else:
                x0 = np.random.uniform(0.1 * max_layer, max_layer, n_var)
            
            x0 = np.clip(x0, 0.0, max_layer)
            s0 = np.sum(x0)
            if s0 > max_total:
                x0 = x0 * (max_total / s0)
                x0 = np.clip(x0, 0.0, max_layer)
            
            # Simulated Annealing
            T = 30.0
            alpha = 0.997
            x_curr = x0.copy()
            curr_val = objective(x_curr)
            
            # Determine steps based on time budget per restart
            time_per_restart = 25.0
            steps_sa = 600
            
            for step in range(steps_sa):
                if time.time() > DEADLINE - 20:
                    break
                
                x_new = x_curr.copy()
                n_perturb = np.random.randint(1, 4)
                idx = np.random.choice(n_var, n_perturb, replace=False)
                
                # Adaptive step size
                step_size = 30.0 * (T / 30.0) + 2.0
                x_new[idx] += np.random.uniform(-step_size, step_size, n_perturb)
                
                x_new = np.clip(x_new, 0.0, max_layer)
                s_new = np.sum(x_new)
                if s_new > max_total:
                    x_new = x_new * (max_total / s_new)
                    x_new = np.clip(x_new, 0.0, max_layer)
                
                new_val = objective(x_new)
                delta_val = new_val - curr_val
                
                if delta_val < 0 or np.random.random() < np.exp(-delta_val / max(T, 1e-9)):
                    x_curr = x_new
                    curr_val = new_val
                    
                T *= alpha
            
            # L-BFGS-B Polish
            from scipy.optimize import minimize
            bounds = [(0.0, max_layer)] * n_var
            # Note: L-BFGS-B does not support linear constraints directly in the same way, 
            # but bounds are handled. The sum constraint is handled by the wrapper projection 
            # inside the objective, which makes the objective non-smooth but often works.
            # Alternatively, we can just run L-BFGS-B on the projected space or rely on the wrapper.
            # Since the wrapper projects, the gradient might be slightly off, but it's a heuristic.
            
            try:
                res = minimize(objective, x_curr, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                if res.fun < curr_val:
                    x_curr = np.clip(res.x, 0.0, max_layer)
                    curr_val = objective(x_curr)
            except Exception:
                pass
            
            # Nelder-Mead Polish (more robust to non-smoothness from projection)
            if time.time() < DEADLINE - 10:
                try:
                    res2 = minimize(objective, x_curr, method='Nelder-Mead',
                                    options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
                    if res2.fun < curr_val:
                        x_curr = np.clip(res2.x, 0.0, max_layer)
                        curr_val = objective(x_curr)
                except Exception:
                    pass
            
            if curr_val < best_val:
                best_val = curr_val
                best_x = x_curr.copy()
        
        if best_x is None:
            return None
        
        return labels, list(best_x), best_val
    
    # Try several layer counts
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Prioritize higher layer counts, but ensure we try a few
    # 12 and 14 are often sweet spots for 400-800nm with 1000nm budget
    layer_counts = [12, 14, 10, 16, 8, 18]
    
    for n_layers in layer_counts:
        if time.time() >= DEADLINE - 10:
            break
        res = optimize_stack(n_layers, n_restarts=3)
        if res is None:
            continue
        labels, th, val = res
        # Validate
        if len(labels) > max_layers:
            continue
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in th):
            continue
        
        # Final projection and check
        th_np = np.array(th)
        th_np = np.clip(th_np, 0.0, max_layer)
        s = np.sum(th_np)
        if s > max_total:
            th_np = th_np * (max_total / s)
            th_np = np.clip(th_np, 0.0, max_layer)
        
        th_list = list(th_np)
        if any(t > max_layer for t in th_list):
            continue
        
        val2 = compute_reflectance(labels, th_list)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th_list
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
