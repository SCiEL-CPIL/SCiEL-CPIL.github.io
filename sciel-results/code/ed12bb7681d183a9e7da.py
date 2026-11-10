"""Combined design program: vectorized AR stack evaluation + Nelder-Mead refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using vectorized evaluation + Nelder-Mead optimization."""
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
        n_layers = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        n = len(materials)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        # Vectorized computation over all wavelengths
        # For each wavelength, compute the product of layer matrices
        # m_total starts as identity for each wavelength
        # Shape: (n_wl, 2, 2) complex
        m_total = np.zeros((n_wl, 2, 2), dtype=complex)
        m_total[:, 0, 0] = 1.0
        m_total[:, 1, 1] = 1.0
        
        for i in range(n):
            ni = n_layers[i]
            di = thicknesses[i]
            # delta shape: (n_wl,)
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)  # (n_wl,)
            s = np.sin(delta)  # (n_wl,)
            
            # Layer matrix for each wavelength
            # m_layer[i, 0, 0] = c[i], m_layer[i, 0, 1] = 1j*s[i]/ni
            # m_layer[i, 1, 0] = 1j*ni*s[i], m_layer[i, 1, 1] = c[i]
            m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
            m_layer[:, 0, 0] = c
            m_layer[:, 0, 1] = 1j * s / ni
            m_layer[:, 1, 0] = 1j * ni * s
            m_layer[:, 1, 1] = c
            
            # Matrix multiply: m_total @ m_layer for each wavelength
            # Use einsum for batched matrix multiplication
            m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
        
        A = m_total[:, 0, 0]
        B = m_total[:, 0, 1]
        C = m_total[:, 1, 0]
        D = m_total[:, 1, 1]
        
        # Admittance at input
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        R = np.abs(r) ** 2
        
        return float(np.mean(R) * 100.0)
    
    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clip_thicknesses(materials, thicknesses):
        """Clip thicknesses to valid range and ensure total constraint."""
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            # Scale down proportionally
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]
    
    # --- Generate candidate structures with diverse starting points ---
    candidates = []
    
    # Quarter-wave thicknesses
    qw_L = lambda_c / (4.0 * nL)
    qw_H = lambda_c / (4.0 * nH)
    
    # Also try half-wave and other ratios for broadband performance
    # For broadband, we often want thicker layers
    
    # Single layer
    candidates.append(("L", [qw_L]))
    candidates.append(("H", [qw_H]))
    candidates.append(("L", [qw_L * 1.2]))  # slightly off quarter-wave
    
    # Two layers
    candidates.append(("LH", [qw_L, qw_H]))
    candidates.append(("HL", [qw_H, qw_L]))
    candidates.append(("LH", [qw_L * 0.8, qw_H * 1.2]))
    candidates.append(("HL", [qw_H * 0.8, qw_L * 1.2]))
    
    # Three layers
    candidates.append(("LHL", [qw_L, qw_H, qw_L]))
    candidates.append(("HLH", [qw_H, qw_L, qw_H]))
    candidates.append(("LLH", [qw_L, qw_L, qw_H]))
    candidates.append(("HHL", [qw_H, qw_H, qw_L]))
    candidates.append(("LHL", [qw_L * 0.9, qw_H * 1.1, qw_L * 0.9]))
    candidates.append(("HLH", [qw_H * 0.9, qw_L * 1.1, qw_H * 0.9]))
    
    # Four layers
    candidates.append(("LHHL", [qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHL", [qw_H, qw_L, qw_L, qw_H]))
    candidates.append(("LHLH", [qw_L, qw_H, qw_L, qw_H]))
    candidates.append(("HLHL", [qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("LHHL", [qw_L * 0.9, qw_H * 1.1, qw_H * 1.1, qw_L * 0.9]))
    
    # Five layers
    candidates.append(("LHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H]))
    candidates.append(("LHLHL", [qw_L * 0.9, qw_H * 1.1, qw_L * 0.9, qw_H * 1.1, qw_L * 0.9]))
    candidates.append(("HLHLH", [qw_H * 0.9, qw_L * 1.1, qw_H * 0.9, qw_L * 1.1, qw_H * 0.9]))
    
    # Six layers
    candidates.append(("LHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    candidates.append(("LHLHLH", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    candidates.append(("HLHLHL", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Seven layers
    candidates.append(("LHLHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Eight layers
    candidates.append(("LHLHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    candidates.append(("LHLHLHLH", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Nine layers
    candidates.append(("LHLHLHLHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    candidates.append(("HLHLHLHLH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Ten layers
    candidates.append(("LHLHLHLHHL", [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H, qw_L]))
    candidates.append(("HLHLHLHLHH", [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_H]))
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for mat_str, thick in candidates:
        materials = list(mat_str)
        if not is_valid(materials, thick):
            continue
        score = mean_reflectance(materials, thick)
        if score < best_score:
            best_score = score
            best_materials = materials
            best_thicknesses = list(thick)
    
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [qw_L]
    
    # --- Nelder-Mead local refinement ---
    current_materials = best_materials
    current_thicknesses = best_thicknesses
    current_score = best_score
    
    n_layers = len(current_materials)
    
    # Nelder-Mead implementation
    def nelder_mead_optimize(materials, x0, max_iter=2000):
        """Simple Nelder-Mead optimization for thickness vector."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        simplex = [list(x0)]
        for i in range(n):
            point = list(x0)
            delta = 10.0 if abs(x0[i]) < 1.0 else 0.1 * abs(x0[i])
            point[i] += delta
            point = clip_thicknesses(materials, point)
            simplex.append(point)
        
        # Evaluate initial simplex
        fvals = []
        for i in range(n + 1):
            if (time.time() - start_time) > DEADLINE:
                return x0, current_score
            fvals.append(mean_reflectance(materials, simplex[i]))
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if fvals[-1] - fvals[0] < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.zeros(n)
            for i in range(n):
                centroid += np.array(simplex[i])
            centroid /= n
            
            # Reflection
            xr = centroid + alpha * (centroid - np.array(simplex[-1]))
            xr = clip_thicknesses(materials, xr)
            fr = mean_reflectance(materials, xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (np.array(xr) - centroid)
                xe = clip_thicknesses(materials, xe)
                fe = mean_reflectance(materials, xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (np.array(xr) - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (np.array(xr) - centroid)
                xc = clip_thicknesses(materials, xc)
                fc = mean_reflectance(materials, xc)
                
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = clip_thicknesses(materials, simplex[i])
                        fvals[i] = mean_reflectance(materials, simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
    
    # Run Nelder-Mead on the best candidate
    x0 = list(current_thicknesses)
    x_opt, score_opt = nelder_mead_optimize(current_materials, x0)
    
    if score_opt < current_score:
        current_thicknesses = x_opt
        current_score = score_opt
    
    # Final validation and clamping
    current_thicknesses = clip_thicknesses(current_materials, current_thicknesses)
    
    if not is_valid(current_materials, current_thicknesses):
        return best_materials, best_thicknesses
    
    return current_materials, current_thicknesses
# EVOLVE-BLOCK-END
