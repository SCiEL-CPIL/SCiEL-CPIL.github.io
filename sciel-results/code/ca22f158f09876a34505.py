"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum(thicknesses, materials, n_amb, n_sub, band_nm, n_wl=401):
    """Compute mean reflectance using characteristic matrix method."""
    wl = np.linspace(band_nm[0], band_nm[1], n_wl)
    n_L = materials['L']
    n_H = materials['H']
    
    # Map materials labels to indices
    n_layers = len(thicknesses)
    n_mat = np.zeros(n_layers)
    for i, mat in enumerate(materials):
        if mat == 'L':
            n_mat[i] = n_L
        else:
            n_mat[i] = n_H
    
    # Optical thicknesses (n*d)
    nd = n_mat * thicknesses
    
    # Phase terms
    k = 2 * np.pi / wl
    phi = k * nd  # shape (n_wl, n_layers)
    
    # Characteristic matrices for each layer
    # M_j = [[cos(phi_j), i*sin(phi_j)/n_j], [i*n_j*sin(phi_j), cos(phi_j)]]
    # We compute the product M = M_1 * M_2 * ... * M_N
    
    # For efficiency, compute using vectorized operations where possible
    # But matrix multiplication is sequential, so we'll do it layer by layer
    
    # Initialize total matrix as identity
    # M_total shape: (n_wl, 2, 2)
    c = np.ones(n_wl, dtype=complex)
    s = np.zeros(n_wl, dtype=complex)
    
    # We'll track M = [[m11, m12], [m21, m22]]
    m11 = np.ones(n_wl, dtype=complex)
    m12 = np.zeros(n_wl, dtype=complex)
    m21 = np.zeros(n_wl, dtype=complex)
    m22 = np.ones(n_wl, dtype=complex)
    
    for j in range(n_layers):
        cj = np.cos(phi[:, j])
        sj = np.sin(phi[:, j])
        nj = n_mat[j]
        
        # M_j = [[cj, 1j*sj/nj], [1j*nj*sj, cj]]
        # M_new = M_old * M_j
        # [[m11, m12], [m21, m22]] * [[cj, 1j*sj/nj], [1j*nj*sj, cj]]
        
        nm11 = m11 * cj + m12 * (1j * sj / nj)
        nm12 = m11 * (1j * sj / nj) + m12 * cj
        nm21 = m21 * cj + m22 * (1j * nj * sj)
        nm22 = m21 * (1j * nj * sj) + m22 * cj
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    # Admittance: Y = (n_sub * m11 + 1j * n_sub * m21) / (n_sub * m12 + 1j * n_sub * m22)
    # Wait, the formula is: Y = (B/C) where [B; C] = M * [1; n_sub]
    # So B = m11 * 1 + m12 * n_sub
    #    C = m21 * 1 + m22 * n_sub
    # Y = B / C
    
    B = m11 + m12 * n_sub
    C = m21 + m22 * n_sub
    Y = B / C
    
    # Reflectance: R = |(Y - n_amb) / (Y + n_amb)|^2
    R = np.abs((Y - n_amb) / (Y + n_amb)) ** 2
    R = np.real(R)
    
    return np.mean(R) * 100.0  # Return in percent

def _nelder_mead(objective, x0, bounds, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-8):
    """Nelder-Mead optimization."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    # Initialize simplex
    simplex = [x.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x.copy()
        simplex[i + 1][i] += 0.05 * max(abs(x[i]), 1.0)
    
    # Evaluate objective
    f = [objective(v) for v in simplex]
    
    for iteration in range(max_iter):
        # Sort simplex
        order = np.argsort(f)
        simplex = [simplex[i] for i in order]
        f = [f[i] for i in order]
        
        # Check convergence
        if abs(f[0] - f[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
        elif fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
        else:
            # Contraction
            if fr < f[-1]:
                xc = centroid + sigma * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f[i] = objective(simplex[i])
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f[i] = objective(simplex[i])
        
        # Apply bounds
        for i in range(n + 1):
            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
            f[i] = objective(simplex[i])
    
    order = np.argsort(f)
    return simplex[order[0]], f[order[0]]

def design(problem):
    """Design broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Start with a good initial guess: symmetric quarter-wave stack
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Initial design: L H L H L (5 layers) or similar
    # Let's use a graded index approach with binary search to approximate continuous profile
    
    # Strategy: Use 20 layers with indices alternating L/H in a pattern that approximates a gradient
    # Then optimize thicknesses
    
    # For now, let's try a multi-start Nelder-Mead on thicknesses with fixed material pattern
    # Material pattern: L H L H ... (10 L and 10 H) or similar
    
    # Let's try several material patterns
    patterns = [
        ['L'] * 20,
        ['H'] * 20,
        ['L', 'H'] * 10,
        ['H', 'L'] * 10,
        ['L', 'L', 'H', 'H'] * 5,
        ['L', 'H', 'H', 'L'] * 5,
        ['L', 'L', 'L', 'H', 'H', 'H'] * 3 + ['L', 'L'],
        ['H', 'H', 'H', 'L', 'L', 'L'] * 3 + ['H', 'H'],
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        n_layers = len(pattern)
        
        # Initial thicknesses: quarter-wave at center wavelength
        init_thick = np.array([center_wl / (4.0 * n_L) if m == 'L' else center_wl / (4.0 * n_H) 
                              for m in pattern])
        
        # Scale to fit within total budget
        total = np.sum(init_thick)
        if total > max_total_nm:
            init_thick *= max_total_nm / total
        
        bounds = (0.0, max_layer_nm)
        
        def objective(thick):
            # Check total thickness constraint
            if np.sum(thick) > max_total_nm + 1e-6:
                return 1000.0
            return _reflectance_spectrum(thick, pattern, n_amb, n_sub, band_nm)
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, init_thick, bounds, max_iter=150)
            if f_opt < best_score:
                best_score = f_opt
                best_materials = pattern
                best_thicknesses = x_opt
        except Exception:
            pass
    
    # Fallback: if no valid solution found, return a simple design
    if best_materials is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * n_L)]
    
    # Ensure thicknesses are within bounds
    best_thicknesses = np.clip(best_thicknesses, 0.0, max_layer_nm)
    
    # Check total thickness
    if np.sum(best_thicknesses) > max_total_nm:
        best_thicknesses *= max_total_nm / np.sum(best_thicknesses)
    
    return best_materials, list(best_thicknesses)
# EVOLVE-BLOCK-END
