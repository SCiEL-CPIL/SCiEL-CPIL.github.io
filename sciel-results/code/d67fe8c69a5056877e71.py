<<<<<<< SEARCH
def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0]
    
    initial_configs = [
        _initial_lattice(),
        _initial_zigzag(),
        _initial_circular(),
        _initial_random(),
        _initial_random(),
    ]
    
    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break
        
        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break
            
            positions = [list(p) for p in init_pos]
            
            result, side = _try_arrangement(positions, oa)
            
            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
=======
def construct():
    """Main entry point using Basin Hopping."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Initial configuration: Hexagonal Lattice
    # Angles in degrees in _initial_lattice, convert to radians for optimization
    init_lattice_deg = _initial_lattice()
    x0_rad = np.array([p for pos in init_lattice_deg for p in pos])
    x0_rad[:, 2] = np.radians(x0_rad[:, 2])
    
    # Run Basin Hopping
    # We use L-BFGS-B for the local minimizer within Basin Hopping
    try:
        bh_result = basinhopping(
            _objective_basin, 
            x0_rad, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'maxfun': 1000}},
            niter=50, # Number of global steps
            stepsize=0.5, # Step size for random perturbations
            seed=42
        )
        best_x = bh_result.x
    except Exception as e:
        print(f"Basin Hopping failed: {e}")
        best_x = x0_rad
        
    # Final polish with L-BFGS-B
    try:
        polish_result = minimize(
            _objective_basin, 
            best_x, 
            method='L-BFGS-B', 
            options={'maxiter': 300, 'maxfun': 3000}
        )
        best_x = polish_result.x
    except Exception:
        pass
        
    # Convert back to degrees for output
    best_pos_rad = best_x.reshape(N, 3)
    best_pos_deg = best_pos_rad.copy()
    best_pos_deg[:, 2] = np.degrees(best_pos_rad[:, 2])
    
    # Calculate final side length and validate
    side = _min_enclosing_side([tuple(p) for p in best_pos_deg], (0.0, 0.0), 0.0)
    
    if _check_validity([tuple(p) for p in best_pos_deg], (0.0, 0.0), side, 0.0, tol=1e-5):
        best_result = ([tuple(p) for p in best_pos_deg], (0.0, 0.0), side, 0.0)
        best_score = side
    else:
        # If invalid, try to repair or use fallback
        # For now, if invalid, we might return the best valid from a simpler search
        # But let's assume BH finds a valid one. If not, we need a fallback.
        pass

    if best_result is None:
        # Fallback to the initial lattice if optimization failed or produced invalid result
        positions = init_lattice_deg
        side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
        if _check_validity(positions, (0.0, 0.0), side, 0.0, tol=1e-5):
             best_result = (positions, (0.0, 0.0), side, 0.0)
        else:
             # Extremely unlikely fallback
             positions = []
             positions.append([0.0, 0.0, 0.0])
             dist = 3.5
             for k in range(6):
                 a = np.radians(60 * k)
                 positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
             dist2 = 6.0
             for k in range(4):
                 a = np.radians(30 + 60 * k)
                 positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
             side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
             best_result = (positions, (0.0, 0.0), side, 0.0)
             
    return best_result
>>>>>>> REPLACE
