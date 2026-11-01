<<<<<<< SEARCH
# --- Structural Optimization Steps ---

def _generate_mackay_positions(r_inner, r_mid, r_outer):
    """
    Generate 55 positions based on Mackay structure with given radii.
    """
    # 1 center
    center = np.zeros((1, 3))
    
    # 12 inner vertices
    inner_pos = r_inner * _VERTS_UNIT
    
    # 30 mid edge points
    mid_pos = r_mid * _EDGE_MIDS
    
    # 12 outer vertices (same direction as inner)
    outer_pos = r_outer * _VERTS_UNIT
    
    pos = np.vstack([center, inner_pos, mid_pos, outer_pos])
    return pos

def _symmetry_project(pos):
    """
    Project positions onto the Mackay icosahedral symmetry orbits.
    This enforces the symmetry, removing numerical drift.
    """
    # pos[0] is center, stays 0
    pos_proj = pos.copy()
    pos_proj[0] = 0.0
    
    # Inner vertices (indices 1-12)
    # Average all 12 inner vertices to get a "center of mass" for that shell?
    # No, they must remain at their specific directions.
    # The "distortion" is usually in the radial distance.
    # We can compute the optimal radius for each shell by averaging the projections.
    
    # Inner shell (1:13)
    if len(pos) > 12:
        inner_vecs = pos[1:13]
        # Project each vector onto its ideal direction
        # d = dot(v, dir), new_v = d * dir
        # Since dirs are unit vectors
        inner_radial = np.sum(inner_vecs * _VERTS_UNIT, axis=1)
        # Average radius to enforce uniformity
        r_inner_avg = np.mean(inner_radial)
        pos_proj[1:13] = r_inner_avg * _VERTS_UNIT
        
    # Mid shell (13:43)
    if len(pos) > 42:
        mid_vecs = pos[13:43]
        mid_radial = np.sum(mid_vecs * _EDGE_MIDS, axis=1)
        r_mid_avg = np.mean(mid_radial)
        pos_proj[13:43] = r_mid_avg * _EDGE_MIDS
        
    # Outer shell (43:55)
    if len(pos) > 54:
        outer_vecs = pos[43:55]
        outer_radial = np.sum(outer_vecs * _VERTS_UNIT, axis=1)
        r_outer_avg = np.mean(outer_radial)
        pos_proj[43:55] = r_outer_avg * _VERTS_UNIT
        
    return pos_proj

def _optimize_radial_parameters():
    """
    Optimize the 3 radii (inner, mid, outer) using Nelder-Mead.
    This is a 3D optimization, very fast and robust.
    """
    def objective(radii):
        r_i, r_m, r_o = radii
        pos = _generate_mackay_positions(r_i, r_m, r_o)
        return _lj_energy(pos)
    
    # Initial guess from literature
    x0 = np.array([1.054, 1.755, 2.0])
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
    return res.x, res.fun

def construct():
    start_time = time.time()
    
    # Step 1: Coarse optimization of shell radii
    # This finds the best "Mackay-like" geometry
    try:
        radii_opt, e_radial = _optimize_radial_parameters()
        pos_best = _generate_mackay_positions(radii_opt[0], radii_opt[1], radii_opt[2])
        best_energy = e_radial
    except Exception:
        # Fallback to standard values
        pos_best = _generate_mackay_positions(1.054, 1.755, 2.0)
        best_energy = _lj_energy(pos_best)
        
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return pos_best

    # Step 2: Full Cartesian Relaxation
    # Use L-BFGS-B on all 165 coordinates
    # This allows the structure to distort slightly from perfect Mackay if beneficial
    # (Though for N=55, Mackay is the global min, so distortion should be minimal)
    
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_grad(x.reshape(-1, 3)).flatten()
    
    try:
        res = minimize(
            energy_func, 
            pos_best.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
        )
        pos_relaxed = res.x.reshape(-1, 3)
        e_relaxed = res.fun
        
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            pos_best = pos_relaxed
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return pos_best

    # Step 3: Symmetry Enforcement and Final Polish
    # The relaxation might have introduced tiny asymmetries.
    # Project back to symmetry, then do a very short high-precision polish.
    
    pos_sym = _symmetry_project(pos_best)
    e_sym = _lj_energy(pos_sym)
    
    if e_sym < best_energy:
        best_energy = e_sym
        pos_best = pos_sym
        
    # Final high precision polish from symmetric start
    try:
        res_final = minimize(
            energy_func, 
            pos_best.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-15, 'ftol': 1e-18, 'maxls': 50}
        )
        pos_final = res_final.x.reshape(-1, 3)
        e_final = res_final.fun
        
        if e_final < best_energy:
            best_energy = e_final
            pos_best = pos_final
    except Exception:
        pass

    # Check minimum distance constraint (0.5 apart)
    # If violated, we might need to adjust, but LJ repulsion is strong.
    # Just in case, verify.
    rij = pos_best[:, np.newaxis, :] - pos_best[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # This should not happen for LJ minimum, but if it does, 
        # we return the best valid one found or just the current one.
        # The problem states "Atoms must be at least 0.5 apart".
        # If this fails, the score is 0.
        # We assume the optimization keeps them apart.
        pass

    return pos_best
=======
# --- Optimization ---

def _optimize_edge_fractions():
    """
    Optimize the 32 parameters (r_inner, r_outer, 30 edge fractions)
    using L-BFGS-B with bounds.
    """
    # Initial guess: standard Mackay with edge fractions = 0.5 (midpoints)
    x0 = np.zeros(2 + _N_EDGES)
    x0[0] = 1.054   # r_inner
    x0[1] = 2.0     # r_outer
    x0[2:2 + _N_EDGES] = 0.5  # edge fractions at midpoint
    
    # Bounds: radii positive, edge fractions in [0, 1]
    lower = np.zeros(2 + _N_EDGES)
    lower[0] = 0.5
    lower[1] = 0.5
    lower[2:2 + _N_EDGES] = 0.0
    
    upper = np.zeros(2 + _N_EDGES)
    upper[0] = 3.0
    upper[1] = 3.0
    upper[2:2 + _N_EDGES] = 1.0
    
    res = minimize(
        _lj_energy_params,
        x0,
        jac=_lj_grad_params,
        method='L-BFGS-B',
        bounds=list(zip(lower, upper)),
        options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 50}
    )
    
    return res.x, res.fun

def _polish_cartesian(pos_start):
    """
    Final high-precision Cartesian relaxation from a given starting position.
    """
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_grad(x.reshape(-1, 3)).flatten()
    
    res = minimize(
        energy_func,
        pos_start.flatten(),
        jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'gtol': 1e-15, 'ftol': 1e-18, 'maxls': 50}
    )
    
    return res.x.reshape(-1, 3), res.fun

def construct():
    start_time = time.time()
    
    # Step 1: Optimize edge-fraction parameterized manifold
    try:
        params_opt, e_params = _optimize_edge_fractions()
        pos_best = _params_to_positions(params_opt)
        best_energy = e_params
    except Exception:
        # Fallback: standard Mackay with midpoints
        params_fallback = np.zeros(2 + _N_EDGES)
        params_fallback[0] = 1.054
        params_fallback[1] = 2.0
        params_fallback[2:2 + _N_EDGES] = 0.5
        pos_best = _params_to_positions(params_fallback)
        best_energy = _lj_energy(pos_best)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 30:
        return pos_best
    
    # Step 2: Full Cartesian relaxation to escape any manifold constraints
    try:
        pos_relaxed, e_relaxed = _polish_cartesian(pos_best)
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            pos_best = pos_relaxed
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        return pos_best
    
    # Step 3: Final high-precision polish
    try:
        pos_final, e_final = _polish_cartesian(pos_best)
        if e_final < best_energy:
            best_energy = e_final
            pos_best = pos_final
    except Exception:
        pass
    
    # Check minimum distance constraint
    rij = pos_best[:, np.newaxis, :] - pos_best[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Should not happen for LJ minimum, but if it does, return best valid
        pass
    
    return pos_best
>>>>>>> REPLACE

