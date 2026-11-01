<<<<<<< SEARCH
def _optimize_radii(radii, maxiter=5000):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        relaxed = _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    result = minimize(objective, radii, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8})
    return result.x
=======
def _optimize_radii(radii, maxiter=5000):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        relaxed = _relax(pos, maxiter=3000, ftol=1e-11, gtol=1e-9)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    # Use multiple Nelder-Mead passes with decreasing step sizes
    current = radii.copy()
    for i in range(3):
        scale = 0.01 / (i + 1)
        result = minimize(objective, current, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-10,
                                   'initial_simplex': current + scale * np.eye(3)})
        current = result.x
    
    return current
>>>>>>> REPLACE

