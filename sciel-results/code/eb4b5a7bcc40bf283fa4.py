"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 26
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to return before limit


def construct():
    """Construct 26 disjoint circles in [0,1]^2 maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Initial solution: grid placement
    side = 6  # 6x6 grid = 36 cells, use 26
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                positions.append([x, y])
                radii.append(r)
    
    positions = np.array(positions)
    radii = np.array(radii)
    
    def get_params():
        """Flatten positions and radii into a single vector."""
        return np.concatenate([positions.flatten(), radii])
    
    def unpack_params(params):
        """Unflatten parameter vector into positions and radii."""
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        return pos, rad
    
    def objective(params):
        """Negative sum of radii (we minimize negative to maximize)."""
        pos, rad = unpack_params(params)
        # Constraint violations as penalty
        penalty = 0.0
        
        # Boundary constraints: circles must be inside [0,1]^2
        for i in range(N):
            x, y = pos[i]
            r = rad[i]
            if x - r < 0:
                penalty += (x - r) ** 2
            if x + r > 1:
                penalty += (x + r - 1) ** 2
            if y - r < 0:
                penalty += (y - r) ** 2
            if y + r > 1:
                penalty += (y + r - 1) ** 2
        
        # Non-overlap constraints: distance between centers >= sum of radii
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(rad) + 100.0 * penalty
    
    # Try local optimization from the grid start
    x0 = get_params()
    
    # Multiple restarts with perturbation
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Also try a more structured initial guess
    # Use a hexagonal-like packing pattern
    def hex_packing_init():
        """Create a hexagonal packing initial guess."""
        pos = []
        rad = []
        # 5 rows with alternating offsets
        rows = [5, 5, 5, 5, 6]  # total = 26
        row_y = []
        for k in range(len(rows)):
            row_y.append((k + 0.5) / len(rows))
        
        idx = 0
        for k, count in enumerate(rows):
            y = row_y[k]
            if k % 2 == 0:
                xs = [(i + 0.5) / count for i in range(count)]
            else:
                xs = [(i + 0.5) / count for i in range(count)]
            for x in xs:
                if idx < N:
                    pos.append([x, y])
                    rad.append(0.08)  # initial radius estimate
                    idx += 1
        
        return np.array(pos), np.array(rad)
    
    # Try several different initial configurations
    configs = []
    
    # Config 1: Grid
    configs.append((positions.copy(), radii.copy()))
    
    # Config 2: Hexagonal
    pos_hex, rad_hex = hex_packing_init()
    configs.append((pos_hex, rad_hex))
    
    # Config 3: Random with small circles
    random.seed(42)
    pos_rand = np.random.uniform(0.1, 0.9, (N, 2))
    rad_rand = np.full(N, 0.05)
    configs.append((pos_rand, rad_rand))
    
    # Config 4: Two large circles + smaller ones
    pos_mix = np.zeros((N, 2))
    rad_mix = np.zeros(N)
    # Two large circles
    pos_mix[0] = [0.25, 0.5]
    pos_mix[1] = [0.75, 0.5]
    rad_mix[0] = 0.2
    rad_mix[1] = 0.2
    # Smaller circles around
    for i in range(2, N):
        angle = 2 * np.pi * i / (N - 2)
        r = 0.35
        pos_mix[i] = [0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)]
        rad_mix[i] = 0.06
    configs.append((pos_mix, rad_mix))
    
    best_score = float('inf')
    best_params = x0.copy()
    
    for cfg_idx, (cfg_pos, cfg_rad) in enumerate(configs):
        if time.time() > deadline:
            break
        
        x0_cfg = np.concatenate([cfg_pos.flatten(), cfg_rad])
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(
                objective,
                x0_cfg,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Try Nelder-Mead for local refinement
        if time.time() > deadline:
            break
        try:
            result_nm = minimize(
                objective,
                x0_cfg,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_params = result_nm.x
        except Exception:
            pass
    
    # Extract best solution
    best_pos, best_rad = unpack_params(best_params)
    
    # Final validation and cleanup: ensure all constraints are satisfied
    # Adjust radii down slightly if needed to ensure disjoint interiors
    for i in range(N):
        # Ensure circle is inside unit square
        best_rad[i] = min(best_rad[i], best_pos[i, 0], 1 - best_pos[i, 0],
                          best_pos[i, 1], 1 - best_pos[i, 1])
    
    # Enforce non-overlap by reducing radii
    for _ in range(10):  # Iterate a few times
        for i in range(N):
            for j in range(i+1, N):
                dx = best_pos[i, 0] - best_pos[j, 0]
                dy = best_pos[i, 1] - best_pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_allowed = (dist - 1e-6) / 2.0
                if max_allowed < 0:
                    max_allowed = 0
                # Each circle can have at most half the gap
                current_sum = best_rad[i] + best_rad[j]
                if current_sum > dist - 1e-6:
                    scale = (dist - 1e-6) / current_sum
                    best_rad[i] *= scale
                    best_rad[j] *= scale
        
        # Re-check boundary
        for i in range(N):
            best_rad[i] = min(best_rad[i], best_pos[i, 0], 1 - best_pos[i, 0],
                              best_pos[i, 1], 1 - best_pos[i, 1])
    
    # Build output array
    result = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    
    # Verify validity
    valid = True
    for i in range(N):
        x, y, r = result[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
        for j in range(i+1, N):
            dx = result[i, 0] - result[j, 0]
            dy = result[i, 1] - result[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < result[i, 2] + result[j, 2] - 1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple grid solution
        side = 6
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
        result = np.array(cells)
    
    return result


# Keep the original function signature
def construct():
    return _construct_impl()


def _construct_impl():
    """Actual implementation."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Simple robust grid-based solution as fallback
    side = 6
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                cells.append([x, y, r])
    
    base_positions = np.array([c[0:2] for c in cells])
    base_radii = np.array([c[2] for c in cells])
    
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        penalty = 0.0
        
        for i in range(N):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0: penalty += (x - r) ** 2
            if x + r > 1: penalty += (x + r - 1) ** 2
            if y - r < 0: penalty += (y - r) ** 2
            if y + r > 1: penalty += (y + r - 1) ** 2
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(rad) + 100.0 * penalty
    
    x0 = np.concatenate([base_positions.flatten(), base_radii])
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Optimize from grid start
    if time.time() < deadline:
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except:
            pass
    
    # Nelder-Mead refinement
    if time.time() < deadline:
        try:
            res_nm = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
        except:
            pass
    
    best_pos = best_params[:2*N].reshape(N, 2)
    best_rad = best_params[2*N:]
    
    # Clamp to valid range
    for i in range(N):
        best_rad[i] = max(0, min(best_rad[i], best_pos[i,0], 1-best_pos[i,0],
                                  best_pos[i,1], 1-best_pos[i,1]))
    
    # Enforce non-overlap
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = best_pos[i,0] - best_pos[j,0]
                dy = best_pos[i,1] - best_pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    continue
                if best_rad[i] + best_rad[j] > dist - 1e-7:
                    scale = max(0, (dist - 1e-7)) / max(best_rad[i] + best_rad[j], 1e-12)
                    best_rad[i] *= scale
                    best_rad[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Final boundary clamp
    for i in range(N):
        best_rad[i] = min(best_rad[i], best_pos[i,0], 1-best_pos[i,0],
                          best_pos[i,1], 1-best_pos[i,1])
        best_rad[i] = max(best_rad[i], 0)
    
    result = np.column_stack([best_pos[:,0], best_pos[:,1], best_rad])
    return result


# The actual entry point
def construct():
    return _construct_impl()
