"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_continuous(n_indices, thicknesses, n_amb, n_sub, wavelengths):
    """Compute reflectance with continuous refractive indices using characteristic matrix method."""
    n_layers = len(n_indices)
    n_waves = len(wavelengths)
    R = np.zeros(n_waves)
    
    for i, lam in enumerate(wavelengths):
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_indices[j]
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            
            if abs(sd) < 1e-12:
                inv_n_sd = 0.0
            else:
                inv_n_sd = 1.0 / (n_j * sd)
            
            B = n_j * sd
            layer_M = np.array([[cd, 1j * inv_n_sd], [1j * B, cd]], dtype=complex)
            M = M @ layer_M
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        r = (n_amb - Y_in) / (n_amb + Y_in)
        R[i] = np.abs(r) ** 2 * 100.0
    
    return R

def _nelder_mead(objective, x0, bounds, maxiter=200, tol=1e-6):
    """Nelder-Mead with bounds clipping."""
    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.1 * (bounds[i][1] - bounds[i][0]) + 0.01
        point[i] += step
        point = np.clip(point, [b[0] for b in bounds], [b[1] for b in bounds])
        simplex.append(point)
    
    values = [objective(p) for p in simplex]
    
    for it in range(maxiter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[0] - values[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < values[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    values[i] = objective(simplex[i])
    
    best_idx = np.argmin(values)
    return np.array(simplex[best_idx]), values[best_idx]

def design(problem):
    """Design antireflection coating using continuous index profile optimization."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Fallback: simple quarter-wave layer
    fallback_materials = ["L"]
    fallback_thicknesses = [band[0] / (4.0 * n_L)]
    
    best_score = 1e6
    best_n_indices = None
    best_thicknesses = None
    
    # Fixed 20-layer basis: optimize continuous indices and thicknesses
    N = 20
    n_low = n_L  # 1.38
    n_high = n_H  # 2.35
    
    def _objective(params):
        """Objective: mean reflectance with constraint penalty."""
        # params: first N are indices, next N are thicknesses
        n_idx = params[:N]
        d_vals = params[N:]
        
        # Penalty for constraint violations
        penalty = 0.0
        if np.any(n_idx < n_low) or np.any(n_idx > n_high):
            penalty += 10.0 * (np.sum(np.maximum(0, n_idx - n_high)) + np.sum(np.maximum(0, n_low - n_idx)))
        if np.any(d_vals < 0) or np.any(d_vals > max_layer_nm):
            penalty += 10.0 * (np.sum(np.maximum(0, d_vals - max_layer_nm)) + np.sum(np.maximum(0, -d_vals)))
        if np.sum(d_vals) > max_total_nm:
            penalty += 10.0 * (np.sum(d_vals) - max_total_nm)
        
        if penalty > 1.0:
            return 100.0 + penalty
        
        R = _reflectance_continuous(n_idx, d_vals, n_amb, n_sub, wavelengths)
        return np.mean(R) + penalty
    
    # Bounds for parameters
    bounds = [(n_low, n_high)] * N + [(0.0, max_layer_nm)] * N
    
    # Multiple starting points
    rng = np.random.default_rng(42)
    
    # Start 1: Graded index from air to substrate
    n_graded = np.linspace(n_amb + 0.05, n_sub - 0.05, N)
    n_graded = np.clip(n_graded, n_low, n_high)
    d_graded = np.full(N, max_total_nm / N)
    x0_graded = np.concatenate([n_graded, d_graded])
    
    # Start 2: Alternating L/H at quarter-wave
    n_alt = np.array([n_L if i % 2 == 0 else n_H for i in range(N)])
    centre = 0.5 * (band[0] + band[1])
    d_alt = np.array([centre / (4.0 * n) for n in n_alt])
    if np.sum(d_alt) > max_total_nm:
        d_alt = d_alt * (max_total_nm / np.sum(d_alt))
    x0_alt = np.concatenate([n_alt, d_alt])
    
    # Start 3: Random
    n_rand = rng.uniform(n_low, n_high, N)
    d_rand = rng.uniform(0, max_layer_nm / 2, N)
    d_rand = d_rand * (max_total_nm / max(np.sum(d_rand), 1e-6))
    x0_rand = np.concatenate([n_rand, d_rand])
    
    # Start 4: Symmetric graded
    half = N // 2
    n_sym = np.concatenate([np.linspace(n_amb + 0.05, n_sub - 0.05, half), 
                            np.linspace(n_sub - 0.05, n_amb + 0.05, N - half)])
    n_sym = np.clip(n_sym, n_low, n_high)
    d_sym = np.full(N, max_total_nm / N)
    x0_sym = np.concatenate([n_sym, d_sym])
    
    starting_points = [x0_graded, x0_alt, x0_rand, x0_sym]
    
    for x0 in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x0,
                bounds,
                maxiter=150,
                tol=1e-5
            )
            
            # Extract and validate
            n_idx = result_x[:N]
            d_vals = result_x[N:]
            
            # Clip to valid range
            n_idx = np.clip(n_idx, n_low, n_high)
            d_vals = np.clip(d_vals, 0.0, max_layer_nm)
            
            if np.sum(d_vals) > max_total_nm:
                d_vals = d_vals * (max_total_nm / np.sum(d_vals))
            
            # Compute actual score
            R = _reflectance_continuous(n_idx, d_vals, n_amb, n_sub, wavelengths)
            actual_score = np.mean(R)
            
            if actual_score < best_score:
                best_score = actual_score
                best_n_indices = n_idx.copy()
                best_thicknesses = d_vals.copy()
                
        except Exception:
            pass
    
    # Convert continuous indices to discrete L/H labels
    if best_n_indices is not None:
        materials = []
        for n_val in best_n_indices:
            # Assign to nearest material
            if abs(n_val - n_L) <= abs(n_val - n_H):
                materials.append("L")
            else:
                materials.append("H")
        
        # Use the optimized thicknesses
        thicknesses = list(best_thicknesses)
        
        # Verify the discrete version
        n_vals_discrete = np.array([problem["materials"][m] for m in materials])
        R_discrete = _reflectance_continuous(n_vals_discrete, thicknesses, n_amb, n_sub, wavelengths)
        discrete_score = np.mean(R_discrete)
        
        # If discrete is worse than continuous, try rounding differently
        if discrete_score > best_score * 1.05:
            # Try alternative assignment: threshold at midpoint
            mid = 0.5 * (n_L + n_H)
            materials_alt = ["L" if n_val < mid else "H" for n_val in best_n_indices]
            n_vals_alt = np.array([problem["materials"][m] for m in materials_alt])
            R_alt = _reflectance_continuous(n_vals_alt, thicknesses, n_amb, n_sub, wavelengths)
            alt_score = np.mean(R_alt)
            
            if alt_score < discrete_score:
                materials = materials_alt
                discrete_score = alt_score
        
        # Final validation
        if all(0 <= t <= max_layer_nm for t in thicknesses) and sum(thicknesses) <= max_total_nm:
            return materials, thicknesses
        else:
            # Fix constraints
            thicknesses = [min(max(t, 0), max_layer_nm) for t in thicknesses]
            if sum(thicknesses) > max_total_nm:
                scale = max_total_nm / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            return materials, thicknesses
    
    # Fallback to simple quarter-wave
    return fallback_materials, fallback_thicknesses
# EVOLVE-BLOCK-END
