<<<<<<< SEARCH
def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Initial arrangement: Compact cluster
    # Using a known good configuration for 12 hexagons
    # 1 center, 6 surrounding, 5 more on the perimeter
    # Or simply a 3x4 rhombus shifted to center
    # Let's use the 3x4 rhombus
    # Rows of 3, 4, 3, 2? No.
    # Let's use the "flower" of 7 + 5 extra
    # A better initial guess is a grid
    # Hex grid coordinates
    # Let's just use the previous seed's best known configuration but optimized
    
    # Start with the configuration from the current program's seed
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
        [0, -4.33, 0]
    ])
    
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Initial parameters for optimizer
    x0 = np.zeros(25)
    x0[:24] = inner[:, :2].flatten()
    x0[24] = 0.0 # Initial outer angle
    
    # Use Nelder-Mead for robustness without gradients
    # It might be slow, so we limit iterations
    best_params = x0.copy()
    best_val = _penalty_function(x0)
    
    # Run multiple restarts with different initial angles and small perturbations
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        if restart == 0:
            current_x0 = x0.copy()
        else:
            current_x0 = x0 + np.random.normal(0, 0.5, 25)
            current_x0[24] = np.random.uniform(0, 60)
            
        try:
            res = minimize(_penalty_function, current_x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Extract best solution
    best_centers = best_params[:24].reshape(12, 2)
    best_outer_angle = best_params[24]
    
    best_inner = np.zeros((12, 3))
    best_inner[:, 0] = best_centers[:, 0]
    best_inner[:, 1] = best_centers[:, 1]
    best_inner[:, 2] = 0.0
    
    best_s = _compute_s(best_inner, best_outer_angle)
    
    # Final validation and cleanup
    best_outer = np.array([0, 0, best_outer_angle])
    
    if not _is_valid(best_inner, best_outer, best_s):
        # Fallback to seed if optimization failed
        inner_seed = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        com_seed = np.mean(inner_seed[:, :2], axis=0)
        inner_seed[:, :2] -= com_seed
        best_inner = inner_seed
        best_outer = np.array([0, 0, 0])
        best_s = _compute_s(best_inner, 0)
        
    return best_inner, best_outer, best_s
=======
def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Initial arrangement: Compact cluster
    # Using a known good configuration for 12 hexagons
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
        [0, -4.33, 0]
    ])
    
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Initial parameters for optimizer
    x0 = np.zeros(25)
    x0[:24] = inner[:, :2].flatten()
    x0[24] = 0.0 # Initial outer angle
    
    # Use Nelder-Mead for robustness without gradients
    best_params = x0.copy()
    best_val = _penalty_function(x0)
    
    # Run multiple restarts with different initial angles and small perturbations
    np.random.seed(42)
    
    # Try different methods and restarts
    methods = ['Nelder-Mead', 'Powell']
    restarts_per_method = 5
    
    for method in methods:
        for restart in range(restarts_per_method):
            if time.time() > deadline:
                break
                
            # Perturb initial guess
            if restart == 0:
                current_x0 = x0.copy()
            else:
                current_x0 = x0 + np.random.normal(0, 0.5, 25)
                current_x0[24] = np.random.uniform(0, 60)
            
            try:
                if method == 'Nelder-Mead':
                    res = minimize(_penalty_function, current_x0, method='Nelder-Mead', 
                                   options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
                else:
                    res = minimize(_penalty_function, current_x0, method='Powell', 
                                   options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
            
        if time.time() > deadline:
            break
            
    # Extract best solution
    best_centers = best_params[:24].reshape(12, 2)
    best_outer_angle = best_params[24]
    
    best_inner = np.zeros((12, 3))
    best_inner[:, 0] = best_centers[:, 0]
    best_inner[:, 1] = best_centers[:, 1]
    best_inner[:, 2] = 0.0
    
    best_s = _compute_s(best_inner, best_outer_angle)
    
    # Final validation and cleanup
    best_outer = np.array([0, 0, best_outer_angle])
    
    if not _is_valid(best_inner, best_outer, best_s):
        # Fallback to seed if optimization failed
        inner_seed = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        com_seed = np.mean(inner_seed[:, :2], axis=0)
        inner_seed[:, :2] -= com_seed
        best_inner = inner_seed
        best_outer = np.array([0, 0, 0])
        best_s = _compute_s(best_inner, 0)
        
    return best_inner, best_outer, best_s
>>>>>>> REPLACE
