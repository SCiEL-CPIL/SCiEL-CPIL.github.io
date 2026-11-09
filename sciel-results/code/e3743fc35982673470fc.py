import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def design(problem):
    """
    Design a broadband antireflection coating using a stack of alternating
    high and low index layers optimized via Differential Evolution.
    """
    t_start = time.time()
    DEADLINE = 150.0 - 5.0  # Leave 5s safety margin

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
        """
        Exact characteristic-matrix method for normal incidence, lossless non-dispersive.
        Returns mean reflectance in percent over the band.
        """
        n_layers = len(labels)
        if n_layers == 0:
            r = (ambient - substrate) / (ambient + substrate)
            R = np.abs(r)**2
            return float(np.mean(R) * 100.0)
            
        # Initialize the characteristic matrix M = [[1,0],[0,1]]
        M00 = np.ones(n_wl, dtype=complex)
        M01 = np.zeros(n_wl, dtype=complex)
        M10 = np.zeros(n_wl, dtype=complex)
        M11 = np.ones(n_wl, dtype=complex)
        
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d == 0:
                continue
            
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            
            s_safe = np.where(np.abs(s) < 1e-12, 1e-12, s)
            
            m00 = c
            m01 = 1j / (n_j * s_safe)
            m10 = 1j * n_j * s_safe
            m11 = c
            
            new_M00 = M00 * m00 + M01 * m10
            new_M01 = M00 * m01 + M01 * m11
            new_M10 = M10 * m00 + M11 * m10
            new_M11 = M10 * m01 + M11 * m11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
            
        Y_sub = substrate
        numerator = M00 + M01 * Y_sub
        denominator = M10 + M11 * Y_sub
        
        denom_safe = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        Y = numerator / denom_safe
        
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)

    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    layer_counts = [4, 6, 8, 10, 12, 14, 16]
    
    for n_layers in layer_counts:
        if time.time() > DEADLINE:
            break
            
        # Determine labels
        labels_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        labels_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        
        for labels in [labels_L, labels_H]:
            if time.time() > DEADLINE:
                break
                
            # Calculate tighter upper bound
            upper_bound = min(max_layer, max_total / n_layers)
            
            def objective(params):
                if time.time() > DEADLINE:
                    return 1000.0
                    
                if any(t < 0 for t in params):
                    return 1000.0
                if any(t > max_layer for t in params):
                    return 1000.0
                if sum(params) > max_total:
                    return 1000.0
                
                return compute_reflectance(labels, params)

            bounds = [(0, upper_bound) for _ in range(n_layers)]
            
            # Differential Evolution
            # Reduce maxiter for higher dimensions to fit in time
            maxiter = max(10, int(100 * (8.0 / n_layers)))
            
            try:
                res = differential_evolution(
                    objective, 
                    bounds=bounds, 
                    seed=42, 
                    maxiter=maxiter, 
                    popsize=15,
                    tol=1e-4,
                    polish=False,
                    updating='immediate'
                )
                
                if res.fun < best_score:
                    if all(0 <= t <= max_layer for t in res.x) and sum(res.x) <= max_total:
                        best_score = res.fun
                        best_labels = labels
                        best_thick = list(res.x)
                        
                        # Polish with L-BFGS-B if time permits
                        if time.time() < DEADLINE - 10:
                            try:
                                res_lbfgs = minimize(
                                    objective, 
                                    res.x, 
                                    method='L-BFGS-B',
                                    bounds=bounds,
                                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                                )
                                if res_lbfgs.fun < best_score:
                                    if all(0 <= t <= max_layer for t in res_lbfgs.x) and sum(res_lbfgs.x) <= max_total:
                                        best_score = res_lbfgs.fun
                                        best_thick = list(res_lbfgs.x)
                            except Exception:
                                pass
            except Exception as e:
                pass

    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
        
    return best_labels, best_thick
