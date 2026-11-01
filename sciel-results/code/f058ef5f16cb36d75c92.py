<<<<<<< SEARCH
def _is_valid_move(arr, i, dx, dy, dr):
    """Check if a move is valid."""
    arr_try = arr.copy()
    arr_try[i, 0] += dx
    arr_try[i, 1] += dy
    arr_try[i, 2] = max(1e-6, arr_try[i, 2] + dr)
    return _is_valid(arr_try)
=======
def _is_valid_move(arr, i, dx, dy, dr):
    """Check if a move is valid."""
    arr_try = arr.copy()
    arr_try[i, 0] += dx
    arr_try[i, 1] += dy
    arr_try[i, 2] = max(1e-6, arr_try[i, 2] + dr)
    return _is_valid(arr_try)


def _slsqp_optimize(arr, time_budget=5.0):
    """SLSQP joint optimization over positions and radii."""
    start_time = time.time()
    n = arr.shape[0]
    
    # Flatten: [x0, y0, r0, x1, y1, r1, ...]
    x0 = arr.flatten()
    
    # Bounds: positions in [0, 1], radii in [0.01, 0.3]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.3))  # r
    
    # Constraints
    constraints = [
        # Perimeter constraint: w + h <= 2
        {
            'type': 'ineq',
            'fun': lambda v: 2.0 - (
                (np.max(v[::3] - v[2::3]) - np.min(v[::3] - v[2::3])) +
                (np.max(v[1::3] - v[2::3]) - np.min(v[1::3] - v[2::3]))
            )
        }
    ]
    
    # Add pairwise non-overlap constraints
    i_idx, j_idx = np.triu_indices(n, k=1)
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i, j=j: (
                np.sqrt((v[i*3] - v[j*3])**2 + (v[i*3+1] - v[j*3+1])**2) -
                (v[i*3+2] + v[j*3+2])
            )
        })
    
    # Objective: maximize sum of radii
    def objective(v):
        return -np.sum(v[2::3])
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result.success and _is_valid(result.x.reshape(n, 3)):
            return result.x.reshape(n, 3), np.sum(result.x[2::3])
        
    except Exception:
        pass
    
    return arr, np.sum(arr[:, 2])
>>>>>>> REPLACE

