"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    total_r = 0.0
    for lam in lambdas:
        # Build total characteristic matrix
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i, mat in enumerate(materials):
            n_i = n_dict[mat]
            d_i = thicknesses[i]
            # Phase thickness delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix:
            # [cos_d, i*sin_d/n_i]
            # [i*n_i*sin_d, cos_d]
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            # Multiply current total matrix M by layer matrix M_i
            # New M = M * M_i
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
            
        # Input admittance Y_in
        Y_in = (C + D * ns) / (A + B * ns)
        
        # Reflectance R = | (Y0 - Y_in) / (Y0 + Y_in) |^2
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _evaluate_candidate(materials, thicknesses, problem):
    """Check constraints and return reflectance or infinity if invalid."""
    if not materials:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
        
    if len(materials) > problem["max_layers"]:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thick = 0.0
    for i, mat in enumerate(materials):
        if mat not in problem["materials"]:
            return float('inf')
        d = thicknesses[i]
        if d < 0 or d > problem["max_layer_nm"]:
            return float('inf')
        total_thick += d
        
    if total_thick > problem["max_total_nm"]:
        return float('inf')
        
    return _mean_reflectance(materials, thicknesses, problem)

def design(problem):
    """Design a stack by numerically optimizing thicknesses for several topologies."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Define topologies to optimize
    # Using alternating L/H structures which are standard for AR coatings
    topologies = [
        ["L", "H", "L", "H"],
        ["H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L", "H", "L"],
    ]
    
    best_materials = None
    best_thicknesses = None
    best_r = float('inf')
    
    # Also keep the simple quarter-wave stack as a fallback/initial guess
    lam_center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    tqL = lam_center / (4.0 * nL)
    tqH = lam_center / (4.0 * nH)
    
    # Evaluate simple quarter wave stacks first to have a baseline
    simple_candidates = []
    for top in topologies:
        thicks = []
        for m in top:
            if m == 'L':
                thicks.append(tqL)
            else:
                thicks.append(tqH)
        r = _evaluate_candidate(top, thicks, problem)
        if r < best_r:
            best_r = r
            best_materials = top
            best_thicknesses = thicks
    
    # Now optimize each topology
    # We use differential_evolution for global optimization
    # Bounds for each thickness: [0, max_layer_nm]
    # However, we must ensure sum <= max_total_nm. 
    # DE doesn't support linear constraints directly. We use penalty in objective.
    
    def objective(thicks, top, prob):
        # Check individual bounds (DE enforces bounds, but just in case)
        # Check sum constraint
        if sum(thicks) > prob["max_total_nm"]:
            return 100.0 # Penalty
        return _mean_reflectance(top, thicks, prob)

    # DE parameters: 
    # popsize: 10-20 is good for small dimensions
    # maxiter: 100-200
    # tol: 1e-4
    # We have 4-8 dimensions. 20*dim evaluations per generation.
    # 100 generations * 20 * 8 = 16000 evaluations.
    # Each evaluation is ~401 wavelengths * 8 layers. Fast.
    # 16000 * 401 * 8 * ~10 flops = ~500M flops. Should be < 1s in numpy/python if optimized, but python loops are slow.
    # The inner loop in _mean_reflectance is python. 16000 evals * 401 waves = 6.4M python iterations.
    # This might be too slow (60s budget).
    # Let's reduce maxiter or popsize, or use a faster local optimizer after a rough DE.
    
    # Strategy: 
    # 1. Run DE with small popsize (15) and maxiter (50) to get a good starting point.
    # 2. Use scipy.optimize.minimize (L-BFGS-B) from that point for fine tuning.
    
    for top in topologies:
        n_layers = len(top)
        bounds = [(0, max_layer_nm) for _ in range(n_layers)]
        
        # Initial guess: quarter wave
        x0 = []
        for m in top:
            if m == 'L':
                x0.append(tqL)
            else:
                x0.append(tqH)
        
        # DE
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                args=(top, problem),
                maxiter=50, 
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False # We will polish manually
            )
            
            # Polish with L-BFGS-B
            from scipy.optimize import minimize
            
            # Ensure initial point is valid
            x_start = res.x
            
            # Check sum constraint for L-BFGS-B bounds? 
            # L-BFGS-B doesn't handle linear equality/inequality well either.
            # We can just minimize the objective which includes penalty.
            
            res2 = minimize(
                objective, 
                x_start, 
                args=(top, problem),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if res2.success or res2.fun < res.fun:
                x_opt = res2.x
            else:
                x_opt = res.x
            
            # Final check and update best
            r_opt = _evaluate_candidate(top, x_opt, problem)
            if r_opt < best_r:
                best_r = r_opt
                best_materials = top
                best_thicknesses = x_opt
                
        except Exception:
            pass
            
    # Fallback if optimization failed or didn't improve
    if best_materials is None:
        best_materials = ["L", "H", "L", "H"]
        best_thicknesses = [tqL, tqH, tqL, tqH]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
