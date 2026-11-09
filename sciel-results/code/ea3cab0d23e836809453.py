"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    # Use a symmetric stack: LH (H-L)...(L-H) reversed -> classic broadband AR
    # Try a few layer counts and pick the best via optimization
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    
    def compute_reflectance(labels, thicknesses_nm):
        """Exact characteristic-matrix method for normal incidence, lossless non-dispersive.
        Returns mean reflectance in percent over the band."""
        n_layers = len(labels)
        # Build per-wavelength matrices and chain
        # M = M1 * M2 * ... * Mn
        # For lossless: M_j = [[cos(delta), (i/sin(delta))/n_j], [i*n_j*sin(delta), cos(delta)]]
        # Use complex arithmetic to be safe
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            # delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_j * d / wavelengths  # shape (n_wl,)
            c = np.cos(delta)          # real-ish
            s = np.sin(delta)
            i_c = 1j
            # per-wavelength 2x2 matrices:
            # M_j[0,0] = c
            # M_j[0,1] = i_c / (n_j * s)
            # M_j[1,0] = i_c * n_j * s
            # M_j[1,1] = c
            M00 = c
            M01 = i_c / (n_j * s)
            M10 = i_c * n_j * s
            M11 = c
            # Multiply M = M * M_j  (vector as column, matrices act on left)
            # We'll accumulate by looping: M_new = M @ M_j
            # But M is 2x2 per wavelength, vectorized:
            new00 = M[0,0]*M00 + M[0,1]*M10
            new01 = M[0,0]*M01 + M[0,1]*M11
            new10 = M[1,0]*M00 + M[1,1]*M10
            new11 = M[1,0]*M01 + M[1,1]*M11
            M = np.array([[new00, new01],[new10, new11]])
        # Admittance: Y = (m11 + m12*Y_sub)/(m21 + m22*Y_sub) with Y_sub = n_sub (normal incidence, lossless)
        Y_sub = substrate
        Y = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        # Reflectance: R = |(Y - n_amb)/(Y + n_amb)|^2
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)  # percent
    
    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence: e.g. L H H L L H H L ... Actually standard: 
        Use a half-stack and mirror. For even n, e.g. n=8: half = 4 labels, full = half + reverse(half)
        We'll use half pattern starting with L on air side."""
        if n_layers % 2 != 0:
            n_layers -= 1
        half = n_layers // 2
        # Simple half pattern: alternating L,H starting with L
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def optimize_stack(n_layers):
        labels = make_symmetric_labels(n_layers)
        if len(labels) == 0:
            return None
        n_var = len(labels)
        
        # Initial guess: quarter-wave at band center
        center = 0.5*(lam_min + lam_max)
        x0 = np.zeros(n_var)
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0[i] = center / (4.0 * n_i)
        x0 = np.clip(x0, 0.0, max_layer)
        # Enforce total
        total0 = np.sum(x0)
        if total0 > max_total:
            x0 = x0 * (max_total / total0)
            x0 = np.clip(x0, 0.0, max_layer)
        
        def objective(x):
            x = np.clip(x, 0.0, max_layer)
            # Enforce total via projection if needed (soft)
            s = np.sum(x)
            if s > max_total:
                x = x * (max_total / s)
                x = np.clip(x, 0.0, max_layer)
            return compute_reflectance(labels, list(x))
        
        from scipy.optimize import minimize
        best_x = x0.copy()
        best_val = objective(best_x)
        
        bounds = [(0.0, max_layer)] * n_var
        # Add linear constraint: sum(x) <= max_total
        cons = [{'type': 'ineq', 'fun': lambda x: max_total - np.sum(np.clip(x,0,max_layer))}]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
            if res.success or res.fun < best_val:
                best_x = np.clip(res.x, 0.0, max_layer)
                s = np.sum(best_x)
                if s > max_total:
                    best_x = best_x * (max_total / s)
                best_val = objective(best_x)
        except Exception:
            pass
        
        # Local polishing with Nelder-Mead on reduced set (if time)
        if time.time() < DEADLINE - 30:
            try:
                res2 = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12})
                if res2.fun < best_val:
                    best_x = np.clip(res2.x, 0.0, max_layer)
                    s = np.sum(best_x)
                    if s > max_total:
                        best_x = best_x * (max_total / s)
                    best_val = objective(best_x)
            except Exception:
                pass
        
        return labels, list(best_x), best_val
    
    # Try several layer counts
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    for n_layers in [10, 12, 8, 14, 16, 6]:
        if time.time() >= DEADLINE:
            break
        res = optimize_stack(n_layers)
        if res is None:
            continue
        labels, th, val = res
        # Validate
        if len(labels) > max_layers:
            continue
        if any(t < 0 or t > max_layer for t in th):
            continue
        if sum(th) > max_total:
            # Scale down
            scale = max_total / sum(th)
            th = [t*scale for t in th]
            if any(t > max_layer for t in th):
                continue
        # Recompute to be safe
        val2 = compute_reflectance(labels, th)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
