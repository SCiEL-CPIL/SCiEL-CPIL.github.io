<<<<<<< SEARCH
def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Try outer angles in finer steps
        for outer_angle_deg in np.arange(0, 30.0, 0.25):
            if time.time() > _DEADLINE - 10.0:
                break
            
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
    
    # Try to improve the best layout with local optimization
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        # Find which layout was best
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 20.0:
                break
            
            # Check if this layout matches the best
            match = True
            for i in range(N):
                if abs(positions[i][0] - best_inner[i][0]) > 1e-6 or abs(positions[i][1] - best_inner[i][1]) > 1e-6:
                    match = False
                    break
            if not match:
                continue
            
            # Local optimization with smooth surrogate
            x0 = []
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(math.radians(inner_angle))
            
            # Try different outer angles with finer steps
            outer_angles_to_try = list(np.arange(best_outer_angle - 10, best_outer_angle + 10, 2.0))
            
            for outer_angle_deg in outer_angles_to_try:
                if time.time() > _DEADLINE - 20.0:
                    break
                
                try:
                    # First pass: use smooth objective for better convergence
                    res = minimize(_objective_smooth, x0, args=(positions, outer_angle_deg, 50.0),
                                  method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                    
                    # Second pass: use exact objective for refinement
                    if res.fun < best_side and res.fun < 100:
                        res2 = minimize(_objective, res.x, args=(positions, outer_angle_deg),
                                       method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                        if res2.fun < best_side and res2.fun < 100:
                            new_positions = []
                            new_angles = []
                            for i in range(N):
                                new_positions.append((res2.x[3*i], res2.x[3*i+1]))
                                new_angles.append(math.degrees(res2.x[3*i+2]))
                            
                            all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                            cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                            
                            if side < best_side:
                                best_side = side
                                best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                                best_center = (cx, cy)
                                best_outer_angle = outer_angle_deg
                except Exception:
                    pass
            break
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final high-precision polish pass
    if time.time() < _DEADLINE - 5.0:
        try:
            # Extract current best parameters
            x0 = []
            for i in range(N):
                x0.append(best_inner[i][0])
                x0.append(best_inner[i][1])
                x0.append(math.radians(best_inner[i][2]))
            
            # Polish with tighter tolerances
            res = minimize(_objective, x0, args=([], best_outer_angle),
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_side and res.fun < 100:
                new_positions = []
                new_angles = []
                for i in range(N):
                    new_positions.append((res.x[3*i], res.x[3*i+1]))
                    new_angles.append(math.degrees(res.x[3*i+2]))
                
                all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                    best_center = (cx, cy)
        except Exception:
            pass
    
    return best_inner, best_center, best_side, best_outer_angle
=======
def _nelder_mead_polish(params, outer_angle_deg, max_iter=3000):
    """Nelder-Mead polish for the exact objective."""
    def obj(p):
        return _objective(p, [], outer_angle_deg)
    
    res = minimize(obj, params, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    return res

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Try outer angles in finer steps
        for outer_angle_deg in np.arange(0, 30.0, 0.25):
            if time.time() > _DEADLINE - 10.0:
                break
            
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
    
    # Try to improve the best layout with local optimization using Nelder-Mead
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        # Find which layout was best
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 15.0:
                break
            
            # Check if this layout matches the best
            match = True
            for i in range(N):
                if abs(positions[i][0] - best_inner[i][0]) > 1e-6 or abs(positions[i][1] - best_inner[i][1]) > 1e-6:
                    match = False
                    break
            if not match:
                continue
            
            # Local optimization with Nelder-Mead (handles nonsmooth objective better)
            x0 = []
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(math.radians(inner_angle))
            
            # Try different outer angles with finer steps
            outer_angles_to_try = list(np.arange(best_outer_angle - 5, best_outer_angle + 5, 1.0))
            
            for outer_angle_deg in outer_angles_to_try:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                try:
                    # Use Nelder-Mead for the nonsmooth exact objective
                    res = _nelder_mead_polish(x0, outer_angle_deg, max_iter=2000)
                    
                    if res.fun < best_side and res.fun < 100:
                        new_positions = []
                        new_angles = []
                        for i in range(N):
                            new_positions.append((res.x[3*i], res.x[3*i+1]))
                            new_angles.append(math.degrees(res.x[3*i+2]))
                        
                        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        
                        if side < best_side:
                            best_side = side
                            best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                            best_center = (cx, cy)
                            best_outer_angle = outer_angle_deg
                except Exception:
                    pass
            break
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final high-precision polish pass with Nelder-Mead
    if time.time() < _DEADLINE - 3.0:
        try:
            # Extract current best parameters
            x0 = []
            for i in range(N):
                x0.append(best_inner[i][0])
                x0.append(best_inner[i][1])
                x0.append(math.radians(best_inner[i][2]))
            
            # Polish with Nelder-Mead and tighter tolerances
            res = _nelder_mead_polish(x0, best_outer_angle, max_iter=5000)
            
            if res.fun < best_side and res.fun < 100:
                new_positions = []
                new_angles = []
                for i in range(N):
                    new_positions.append((res.x[3*i], res.x[3*i+1]))
                    new_angles.append(math.degrees(res.x[3*i+2]))
                
                all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                    best_center = (cx, cy)
        except Exception:
            pass
    
    return best_inner, best_center, best_side, best_outer_angle
>>>>>>> REPLACE
