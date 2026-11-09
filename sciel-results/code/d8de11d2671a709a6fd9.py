"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating using a periodic alternating stack
    with linear gradient thickness, optimized via Differential Evolution and L-BFGS-B."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 10.0  # Leave 10s safety margin
    
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
            # Bare substrate
            r = (ambient - substrate) / (ambient + substrate)
            R = np.abs(r)**2
            return float(np.mean(R) * 100.0)
            
        # Pre-allocate M components
        M00 = np.ones(n_wl, dtype=complex)
        M01 = np.zeros(n_wl, dtype=complex)
        M10 = np.zeros(n_wl, dtype=complex)
        M11 = np.ones(n_wl, dtype=complex)
        
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d == 0:
                continue
            # delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            
            # Avoid division by zero if sin(delta) is very small
            s_safe = np.where(np.abs(s) < 1e-12, 1e-12, s)
            
            # Matrix elements
            m00 = c
            m01 = 1j / (n_j * s_safe)
            m10 = 1j * n_j * s_safe
            m11 = c
            
            # M = M @ M_j
            new_M00 = M00 * m00 + M01 * m10
            new_M01 = M00 * m01 + M01 * m11
            new_M10 = M10 * m00 + M11 * m10
            new_M11 = M10 * m01 + M11 * m11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
            
        # Admittance Y = (M11 + M12*Y_sub) / (M21 + M22*Y_sub)
        Y_sub = substrate
        numerator = M00 + M01 * Y_sub
        denominator = M10 + M11 * Y_sub
        
        # Avoid division by zero
        denom_safe = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        Y = numerator / denom_safe
        
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)

    def generate_periodic_gradient_stack(n_layers, base_thick, grad_factor):
        """
        Generate a periodic alternating L/H stack with linear gradient thickness.
        Labels alternate L, H, L, H...
        Thicknesses: t_i = base_thick * (1 + grad_factor * (i - (n_layers-1)/2) / max(1, (n_layers-1)/2))
        This creates a symmetric gradient centered at the middle layer.
        """
        if n_layers <= 0:
            return [], []
        
        labels = []
        thicknesses = []
        
        # Alternate starting with L
        for i in range(n_layers):
            lab = "L" if i % 2 == 0 else "H"
            labels.append(lab)
            
            # Center index
            center_idx = (n_layers - 1) / 2.0
            # Normalized position: -1 to 1
            norm_pos = (i - center_idx) / max(1, center_idx) if center_idx > 0 else 0.0
            t = base_thick * (1.0 + grad_factor * norm_pos)
            
            # Clamp to valid range
            t = max(0.0, min(max_layer, t))
            thicknesses.append(t)
        
        return labels, thicknesses

    def objective(params):
        """
        Objective function for optimization.
        params: [base_thick, grad_factor]
        """
        base_thick = params[0]
        grad_factor = params[1]
        
        nonlocal current_n_layers
        labels, thicknesses = generate_periodic_gradient_stack(current_n_layers, base_thick, grad_factor)
        
        # Check constraints
        if any(t < 0 or t > max_layer for t in thicknesses):
            return 100.0
        if sum(thicknesses) > max_total:
            return 100.0
            
        return compute_reflectance(labels, thicknesses)

    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Try different numbers of layers
    # Periodic alternating stacks work well with even numbers of layers
    for n_layers in [10, 12, 14, 16, 8]:
        if time.time() >= DEADLINE:
            break
            
        current_n_layers = n_layers
        
        # Bounds for base thickness and gradient factor
        center_lambda = 0.5 * (lam_min + lam_max)
        min_base = center_lambda / (8.0 * max(n_L, n_H))
        max_base = center_lambda / (2.0 * min(n_L, n_H))
        
        # Gradient factor bounds: -1 to 1 (relative change)
        bounds = [(min_base, max_base), (-1.0, 1.0)]
        
        # 1. Global Search: Differential Evolution
        try:
            from scipy.optimize import differential_evolution
            if time.time() < DEADLINE - 30:
                res_de = differential_evolution(
                    objective, 
                    bounds=bounds, 
                    seed=42, 
                    maxiter=50, 
                    popsize=15,
                    tol=1e-4,
                    polish=False
                )
                if res_de.fun < best_score:
                    best_score = res_de.fun
                    best_params = res_de.x
                    labels, thicknesses = generate_periodic_gradient_stack(n_layers, best_params[0], best_params[1])
                    if all(0 <= t <= max_layer for t in thicknesses) and sum(thicknesses) <= max_total:
                        best_labels = labels
                        best_thick = thicknesses
        except Exception as e:
            pass
            
        # 2. Local Polish: L-BFGS-B
        # Start from a reasonable guess: base thickness ~ lambda/4 for L, small gradient
        base_guess = center_lambda / (4.0 * n_L)
        grad_guess = 0.2
        
        x0 = [base_guess, grad_guess]
        
        # If DE found something better, use that as start
        if 'best_params' in locals() and res_de.fun < 100.0:
            x0 = res_de.x
            
        try:
            from scipy.optimize import minimize
            res_lbfgs = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_params = res_lbfgs.x
                labels, thicknesses = generate_periodic_gradient_stack(n_layers, best_params[0], best_params[1])
                if all(0 <= t <= max_layer for t in thicknesses) and sum(thicknesses) <= max_total:
                    best_labels = labels
                    best_thick = thicknesses
        except Exception as e:
            pass

    # Fallback to a simple quarter-wave layer if nothing worked
    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
        
    return best_labels, best_thick
# EVOLVE-BLOCK-END
