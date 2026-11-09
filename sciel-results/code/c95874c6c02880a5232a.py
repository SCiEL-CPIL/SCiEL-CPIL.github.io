<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
=======
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial import ConvexHull
>>>>>>> REPLACE
<<<<<<< SEARCH
def _compute_side_length(inner_hexes, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        for i in range(len(inner_hexes)):
            x, y, a = inner_hexes[i]
            v = _hex_vertices(x, y, 1.0, a)
            for p in v:
                dp = p - outer_center
                proj = dp @ n
                if proj > 0:
                    required_s = proj * 2.0 / np.sqrt(3.0)
                    max_s = max(max_s, required_s)
    
    return max_s
=======
def _compute_side_length(inner_hexes, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    # For each of the 6 edge normals of the outer hexagon, find the max projection
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        # Project all points onto the normal
        dps = all_points - outer_center
        projs = dps @ n
        max_proj = projs.max()
        if max_proj > 0:
            required_s = max_proj * 2.0 / np.sqrt(3.0)
            max_s = max(max_s, required_s)
    
    return max_s
>>>>>>> REPLACE
<<<<<<< SEARCH
def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses free-body simulation for initial refinement, then differential evolution and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    outer_center = np.array([0.0, 0.0])
    
    best_params = None
    best_side = float('inf')
    best_angle = 0.0
    
    # Try different outer angles and initial configurations
    angles_to_try = [0, 30, 60, 90, 120, 150]
    
    # Initial symmetric configurations with different spacing
    init_configs = []
    for r1 in [2.0, 2.1, 2.2, 2.3]:
        for r2 in [3.5, 3.8, 4.0, 4.2]:
            init_configs.append(_make_symmetric_init(r1, r2))
    
    # Also add the original grid-based init
    sqrt3 = np.sqrt(3)
    init_configs.append(np.array([
        [0, 0, 0],
        [2.0, 0, 0],
        [1.0, sqrt3, 0],
        [-1.0, sqrt3, 0],
        [-2.0, 0, 0],
        [-1.0, -sqrt3, 0],
        [1.0, -sqrt3, 0],
        [3.0, sqrt3, 0],
        [0, 2*sqrt3, 0],
        [-3.0, sqrt3, 0],
        [3.0, -sqrt3, 0],
    ]))
    
    # Evaluate all initial configs and keep the best, using free-body simulation
    for init in init_configs:
        if time.time() > deadline - 60:
            break
        for angle in angles_to_try[:4]:  # Try more angles for initial evaluation
            if time.time() > deadline - 60:
                break
            # Apply free-body simulation to refine
            refined = _free_body_simulate(init, outer_center, float(angle), max_steps=200, dt=0.02)
            
            side = _compute_side_length(refined, outer_center, float(angle))
            if _all_pairs_separated(refined) and _outer_contains_inner(outer_center, side, float(angle), refined):
                if side < best_side:
                    best_params = refined.copy()
                    best_side = side
                    best_angle = float(angle)
    
    if best_params is None:
        # Fallback to first config
        best_params = init_configs[0]
        best_angle = 0.0
        best_side = _compute_side_length(best_params, outer_center, best_angle)
    
    # Phase 1: Differential evolution for global optimization
    if time.time() < deadline - 120:
        # Bounds for each hexagon: x in [-5,5], y in [-5,5], angle in [0,180]
        bounds = [(-5, 5)] * 11 + [(-5, 5)] * 11 + [(0, 180)] * 11
        
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(outer_center, best_angle),
                maxiter=300,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < best_side:
                candidate = de_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from best found
    if time.time() < deadline - 80:
        try:
            nm_result = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result.fun < best_side:
                candidate = nm_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Phase 3: Free-body simulation from best found for further refinement
    if time.time() < deadline - 40:
        refined = _free_body_simulate(best_params, outer_center, best_angle, max_steps=1000, dt=0.005)
        side = _compute_side_length(refined, outer_center, best_angle)
        if _all_pairs_separated(refined) and _outer_contains_inner(outer_center, side, best_angle, refined):
            if side < best_side:
                best_params = refined
                best_side = side
    
    # Phase 4: Try other outer angles with the best configuration
    if time.time() < deadline - 30:
        for angle in angles_to_try:
            if time.time() > deadline - 15:
                break
            if abs(angle - best_angle) < 1:
                continue
            try:
                side = _compute_side_length(best_params, outer_center, float(angle))
                if _all_pairs_separated(best_params) and _outer_contains_inner(outer_center, side, float(angle), best_params):
                    if side < best_side:
                        best_side = side
                        best_angle = float(angle)
            except Exception:
                continue
    
    # Final refinement with best angle
    if time.time() < deadline - 15:
        try:
            nm_result2 = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result2.fun < best_side:
                candidate = nm_result2.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Final verification and fallback
    final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    if not _all_pairs_separated(best_params) or not _outer_contains_inner(outer_center, final_side, best_angle, best_params):
        # Fallback to a known valid configuration
        sqrt3 = np.sqrt(3)
        best_params = np.array([
            [0, 0, 0],
            [2.0, 0, 0],
            [1.0, sqrt3, 0],
            [-1.0, sqrt3, 0],
            [-2.0, 0, 0],
            [-1.0, -sqrt3, 0],
            [1.0, -sqrt3, 0],
            [3.0, sqrt3, 0],
            [0, 2*sqrt3, 0],
            [-3.0, sqrt3, 0],
            [3.0, -sqrt3, 0],
        ])
        best_angle = 0.0
        final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    outer_hex_side_length = float(final_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
=======
def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses a structural approach: exploit 6-fold symmetry, optimize ring distances and angles.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    outer_center = np.array([0.0, 0.0])
    sqrt3 = np.sqrt(3)
    
    best_params = None
    best_side = float('inf')
    best_angle = 0.0
    
    # The key insight: with 6-fold symmetry, we have:
    # - 1 center hexagon at origin
    # - 6 hexagons at distance r1 from center, at angles 0, 60, 120, 180, 240, 300
    # - 4 hexagons at distance r2 from center, at angles 30, 150, 210, 330 (offset by 30 deg)
    #   But actually, for 11 = 1 + 6 + 4, the 4 outer ones are at specific positions.
    # 
    # Let's parameterize by: r1, r2, theta1 (angle offset for ring 1), theta2 (angle offset for ring 2)
    # and the rotation angles of each hexagon.
    
    def _make_config(r1, r2, theta1, theta2, rot1, rot2, rot0):
        """Create configuration from symmetric parameters."""
        config = np.zeros((11, 3))
        # Center
        config[0] = [0, 0, rot0]
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = np.radians(theta1 + 60 * k)
            config[1 + k, 0] = r1 * np.cos(ang)
            config[1 + k, 1] = r1 * np.sin(ang)
            config[1 + k, 2] = rot1
        # Ring 2: 4 hexagons at positions that fill gaps
        # Place them at angles that are offset from ring 1
        angles2 = [np.radians(theta2 + 150 * k) for k in range(4)]
        # Actually, let's use a different placement: 4 hexagons at the "corners"
        # For 11 hexagons in a hexagonal arrangement, the 4 outer ones are at:
        # positions that form a smaller hexagon with 2 missing
        # Let's try: angles 30, 150, 210, 330 (offset by 30 from ring 1)
        angles2 = [np.radians(30 + 120 * k) for k in range(4)]
        for k in range(4):
            idx = 7 + k
            config[idx, 0] = r2 * np.cos(angles2[k])
            config[idx, 1] = r2 * np.sin(angles2[k])
            config[idx, 2] = rot2
        return config
    
    def _objective_sym(params):
        """Objective for symmetric parameter optimization."""
        r1, r2, theta1, theta2, rot1, rot2, rot0, outer_angle = params
        if r1 < 1.5 or r2 < r1 + 1.0 or r1 > 10 or r2 > 15:
            return 1e6
        
        config = _make_config(r1, r2, theta1, theta2, rot1, rot2, rot0)
        
        if not _all_pairs_separated(config):
            return 1e6
        
        side = _compute_side_length(config, outer_center, outer_angle)
        
        if not _outer_contains_inner(outer_center, side, outer_angle, config):
            return 1e6
        
        return side
    
    # Try various outer angles
    angles_to_try = [0, 30, 60, 90, 120, 150]
    
    # Phase 1: Grid search over symmetric parameters
    for outer_angle in angles_to_try:
        if time.time() > deadline - 100:
            break
        
        # Grid search over key parameters
        best_local = None
        best_local_side = float('inf')
        
        for r1 in np.linspace(1.8, 3.5, 10):
            if time.time() > deadline - 100:
                break
            for r2 in np.linspace(r1 + 1.2, r1 + 4.0, 10):
                if time.time() > deadline - 100:
                    break
                for theta1 in [0, 15, 30]:
                    if time.time() > deadline - 100:
                        break
                    for rot1 in [0, 30, 60]:
                        if time.time() > deadline - 100:
                            break
                        params = [r1, r2, theta1, 30, rot1, rot1, 0, outer_angle]
                        val = _objective_sym(params)
                        if val < best_local_side:
                            best_local_side = val
                            best_local = params
        
        if best_local is not None and best_local_side < best_side:
            # Refine with local optimization
            try:
                nm_result = minimize(
                    _objective_sym,
                    best_local,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                if nm_result.fun < best_local_side:
                    best_local_side = nm_result.fun
                    best_local = nm_result.x
            except Exception:
                pass
            
            if best_local_side < best_side:
                r1, r2, theta1, theta2, rot1, rot2, rot0, outer_angle = best_local
                best_params = _make_config(r1, r2, theta1, theta2, rot1, rot2, rot0)
                best_side = best_local_side
                best_angle = outer_angle
    
    # Phase 2: If we have a good symmetric solution, do full optimization
    if best_params is not None and time.time() < deadline - 60:
        # Use the symmetric solution as initial guess for full optimization
        bounds = [(-5, 5)] * 11 + [(-5, 5)] * 11 + [(0, 180)] * 11
        
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(outer_center, best_angle),
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                x0=best_params.flatten()
            )
            
            if de_result.fun < best_side:
                candidate = de_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Phase 3: Final local refinement
    if best_params is not None and time.time() < deadline - 20:
        try:
            nm_result = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result.fun < best_side:
                candidate = nm_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_params is None:
        # Use a known valid configuration
        best_params = np.array([
            [0, 0, 0],
            [2.0, 0, 0],
            [1.0, sqrt3, 0],
            [-1.0, sqrt3, 0],
            [-2.0, 0, 0],
            [-1.0, -sqrt3, 0],
            [1.0, -sqrt3, 0],
            [3.0, sqrt3, 0],
            [0, 2*sqrt3, 0],
            [-3.0, sqrt3, 0],
            [3.0, -sqrt3, 0],
        ])
        best_angle = 0.0
        best_side = _compute_side_length(best_params, outer_center, best_angle)
    
    # Final verification
    final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    if not _all_pairs_separated(best_params) or not _outer_contains_inner(outer_center, final_side, best_angle, best_params):
        # Fallback to a known valid configuration
        best_params = np.array([
            [0, 0, 0],
            [2.0, 0, 0],
            [1.0, sqrt3, 0],
            [-1.0, sqrt3, 0],
            [-2.0, 0, 0],
            [-1.0, -sqrt3, 0],
            [1.0, -sqrt3, 0],
            [3.0, sqrt3, 0],
            [0, 2*sqrt3, 0],
            [-3.0, sqrt3, 0],
            [3.0, -sqrt3, 0],
        ])
        best_angle = 0.0
        final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    outer_hex_side_length = float(final_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
>>>>>>> REPLACE
