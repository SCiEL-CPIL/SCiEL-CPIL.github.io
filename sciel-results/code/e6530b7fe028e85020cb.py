import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    E = -122880.0 * c3
    F = 4096.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            r = root.real
            if r > 1e-12:
                eps = max(1e-8, abs(r) * 1e-8)
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def _g_values_fast(c0, c1, c2, x_grid):
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    x2 = x_grid ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x4 ** 2
    x10 = x8 * x2
    x12 = x8 * x4
    
    g = (c0 * 1.0 +
         c1 * (16.0 * x4 - 48.0 * x2 + 12.0) +
         c2 * (256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0) +
         c3 * (4096.0 * x12 - 122880.0 * x10 + 1397760.0 * x8 - 6812160.0 * x6 + 13977600.0 * x4 - 11827200.0 * x2 + 4084080.0))
    return g

def _check_positivity(c0, c1, c2, x_grid):
    """Check if g(x) >= 0 for all x in x_grid with a small tolerance."""
    g_vals = _g_values_fast(c0, c1, c2, x_grid)
    min_g = np.min(g_vals)
    return min_g >= -1e-8, min_g

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Create grids for positivity checking
    x_coarse = np.linspace(0.001, 8.0, 500)
    x_fine = np.linspace(0.001, 8.0, 2000)
    
    def objective(params):
        """Objective: minimize value + large penalty for violating positivity."""
        c0, c1, c2 = params
        val = compute_value([c0, c1, c2])
        if val >= 1e8:
            return 1e8
        
        # Check positivity on coarse grid first (fast)
        g_coarse = _g_values_fast(c0, c1, c2, x_coarse)
        min_coarse = np.min(g_coarse)
        
        if min_coarse < -0.1:
            # Strong violation, return large penalty
            return val + 1000.0 * (-min_coarse)
        
        # If close to violation, check on fine grid
        if min_coarse < -1e-4:
            g_fine = _g_values_fast(c0, c1, c2, x_fine)
            min_fine = np.min(g_fine)
            if min_fine < -1e-8:
                return val + 1000.0 * (-min_fine)
        
        return val
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            de_result = differential_evolution(
                objective, 
                bounds, 
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                init='sobol',
                updating='deferred',
                polish=False
            )
            
            # Verify positivity on fine grid
            ok, min_g = _check_positivity(de_result.x[0], de_result.x[1], de_result.x[2], x_fine)
            if ok:
                actual_val = compute_value(de_result.x)
                if actual_val < best_val:
                    best_val = actual_val
                    best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from best point
    if time.time() < deadline - 20.0:
        try:
            def objective_refine(params):
                c0, c1, c2 = params
                val = compute_value([c0, c1, c2])
                if val >= 1e8:
                    return 1e8
                g_vals = _g_values_fast(c0, c1, c2, x_fine)
                min_g = np.min(g_vals)
                if min_g < -1e-8:
                    return val + 100.0 * (-min_g)
                return val
            
            res = minimize(
                objective_refine, 
                best_coeffs, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            ok, min_g = _check_positivity(res.x[0], res.x[1], res.x[2], x_fine)
            if ok:
                actual_val = compute_value(res.x)
                if actual_val < best_val:
                    best_val = actual_val
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try a few random starts with local optimization
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            try:
                start = seed + rng.normal(0, 0.5, size=3)
                res = minimize(
                    objective, 
                    start, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                
                ok, min_g = _check_positivity(res.x[0], res.x[1], res.x[2], x_fine)
                if ok:
                    actual_val = compute_value(res.x)
                    if actual_val < best_val:
                        best_val = actual_val
                        best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Final verification
    if best_val < seed_val:
        ok, min_g = _check_positivity(best_coeffs[0], best_coeffs[1], best_coeffs[2], x_fine)
        if ok:
            final_val = compute_value(best_coeffs)
            if final_val < best_val:
                best_val = final_val
            return best_coeffs
    
    return seed
