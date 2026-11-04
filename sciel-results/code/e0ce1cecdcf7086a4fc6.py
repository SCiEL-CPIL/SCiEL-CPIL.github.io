"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 21
DEADLINE = 135.0  # safety margin before 150s kill


def _check_validity(w, h, xy, r, tol=1e-7):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if w <= 0 or h <= 0:
        return False
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    for i in range(N):
        x, y = xy[i]
        ri = r[i]
        if ri < -tol:
            return False
        if x - ri < -tol or x + ri > w + tol or y - ri < -tol or y + ri > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(w, h, xy, r, tol=1e-7):
    """Return sum of radii if valid, else -inf."""
    if not _check_validity(w, h, xy, r, tol):
        return -np.inf
    return float(np.sum(r))


def _hexagonal_layout(w, h, n_cols, n_rows):
    """
    Generate a hexagonal packing layout for n_cols x n_rows circles.
    Returns (xy, r) where xy is (N,2) and r is (N,).
    """
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # Hexagonal spacing: horizontal spacing = 2*r, vertical spacing = sqrt(3)*r
    # For a given number of columns and rows, we need to fit them in w x h.
    # Let's parameterize by r and compute positions.
    
    # Actually, let's just place circles in a hex grid pattern scaled to fit.
    # We'll use a unit hex grid and scale it.
    
    # Hex grid positions in unit space (r=0.5 for unit spacing)
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(positions) >= N:
                break
            # Hex offset
            x = col + (0.5 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3) / 2
            positions.append([x, y])
    
    positions = np.array(positions[:N])
    
    if len(positions) == 0:
        return xy, r
    
    # Scale to fit in w x h
    # The hex grid has extent: width = (n_cols - 1 + 0.5) * 1, height = (n_rows - 1) * sqrt(3)/2
    extent_x = (n_cols - 1 + 0.5) if n_cols > 1 else 1.0
    extent_y = (n_rows - 1) * np.sqrt(3) / 2 if n_rows > 1 else 1.0
    
    # Minimum radius in unit space is 0.5 (circles of radius 0.5 centered at integer/half-integer positions)
    # We need to scale so that circles of radius r fit.
    # If we scale by factor s, positions become s * positions, radii become s * 0.5
    # Constraint: s * extent_x + 2 * (s * 0.5) <= w  =>  s * (extent_x + 1) <= w
    #             s * extent_y + 2 * (s * 0.5) <= h  =>  s * (extent_y + 1) <= h
    
    scale_x = w / (extent_x + 1.0)
    scale_y = h / (extent_y + 1.0)
    scale = min(scale_x, scale_y)
    
    # Center the layout
    pos_scaled = positions * scale
    center_offset_x = (w - pos_scaled[:, 0].max() + pos_scaled[:, 0].min()) / 2
    center_offset_y = (h - pos_scaled[:, 1].max() + pos_scaled[:, 1].min()) / 2
    
    xy = pos_scaled + np.array([center_offset_x, center_offset_y])
    r = np.full(N, scale * 0.5)
    
    return xy, r


def _grid_layout(w, h, n_cols, n_rows):
    """Generate a grid layout."""
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(positions) >= N:
                break
            x = col
            y = row
            positions.append([x, y])
    
    positions = np.array(positions[:N])
    
    if len(positions) == 0:
        return xy, r
    
    extent_x = n_cols - 1 if n_cols > 1 else 0
    extent_y = n_rows - 1 if n_rows > 1 else 0
    
    scale_x = w / (extent_x + 1.0)
    scale_y = h / (extent_y + 1.0)
    scale = min(scale_x, scale_y)
    
    pos_scaled = positions * scale
    center_offset_x = (w - pos_scaled[:, 0].max() + pos_scaled[:, 0].min()) / 2
    center_offset_y = (h - pos_scaled[:, 1].max() + pos_scaled[:, 1].min()) / 2
    
    xy = pos_scaled + np.array([center_offset_x, center_offset_y])
    r = np.full(N, scale * 0.5)
    
    return xy, r


def _random_layout(w, h):
    """Generate a random layout."""
    xy = np.random.uniform(0.05, 0.95, size=(N, 2)) * np.array([w, h])
    r = np.random.uniform(0.02, 0.08, size=N)
    return xy, r


def _repair_and_optimize(w, h, xy, r, time_budget=5.0):
    """
    Repair a layout to be valid, then optimize radii and positions.
    Uses iterative separation and radius maximization.
    """
    start = time.time()
    xy = xy.copy()
    r = r.copy()
    
    # First, ensure all circles are within bounds
    for i in range(N):
        r[i] = min(r[i], xy[i, 0], w - xy[i, 0], xy[i, 1], h - xy[i, 1])
        r[i] = max(r[i], 0.001)
        xy[i, 0] = np.clip(xy[i, 0], r[i], w - r[i])
        xy[i, 1] = np.clip(xy[i, 1], r[i], h - r[i])
    
    # Iterative optimization: separate overlaps, then grow radii
    best_score = _score(w, h, xy, r)
    
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Separate overlapping circles
        for _ in range(50):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        overlap = min_dist - dist
                        push_x = dx / dist * overlap * 0.5
                        push_y = dy / dist * overlap * 0.5
                        xy[i, 0] += push_x
                        xy[i, 1] += push_y
                        xy[j, 0] -= push_x
                        xy[j, 1] -= push_y
                        moved = True
                    elif dist <= 1e-10:
                        # Random push
                        angle = random.uniform(0, 2 * np.pi)
                        push = min_dist * 0.5
                        xy[i, 0] += np.cos(angle) * push
                        xy[i, 1] += np.sin(angle) * push
                        xy[j, 0] -= np.cos(angle) * push
                        xy[j, 1] -= np.sin(angle) * push
                        moved = True
            
            # Clamp to boundaries
            for i in range(N):
                xy[i, 0] = np.clip(xy[i, 0], r[i], w - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], h - r[i])
            
            if not moved:
                break
        
        # Try to grow radii
        growth = 1.001
        new_r = r * growth
        if _check_validity(w, h, xy, new_r, tol=1e-6):
            r = new_r
        
        current_score = _score(w, h, xy, r)
        if current_score > best_score:
            best_score = current_score
    
    return xy, r, best_score


def _polish(w, h, xy, r, time_budget=10.0):
    """
    Final polish: local optimization of positions and radii.
    Uses gradient-free local search.
    """
    start = time.time()
    xy = xy.copy()
    r = r.copy()
    best_score = _score(w, h, xy, r)
    
    # Local search: perturb positions and radii
    step_size = 0.01
    for _ in range(500):
        if time.time() - start > time_budget:
            break
        
        # Try small perturbations
        for _ in range(20):
            xy_trial = xy.copy()
            r_trial = r.copy()
            
            # Perturb a random circle
            i = random.randint(0, N - 1)
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            xy_trial[i, 0] += dx
            xy_trial[i, 1] += dy
            
            # Try to grow this circle's radius
            r_trial[i] *= 1.005
            
            # Clamp
            xy_trial[i, 0] = np.clip(xy_trial[i, 0], r_trial[i], w - r_trial[i])
            xy_trial[i, 1] = np.clip(xy_trial[i, 1], r_trial[i], h - r_trial[i])
            
            if _check_validity(w, h, xy_trial, r_trial, tol=1e-7):
                trial_score = _score(w, h, xy_trial, r_trial)
                if trial_score > best_score:
                    xy = xy_trial
                    r = r_trial
                    best_score = trial_score
        
        # Gradually decrease step size
        step_size *= 0.99
    
    return xy, r, best_score


def _pack_for_aspect(w, h, time_budget=15.0):
    """
    Pack circles in a rectangle of given dimensions.
    Try multiple initial layouts and optimize.
    """
    start = time.time()
    best_score = -np.inf
    best_xy = None
    best_r = None
    
    # Try different initial layouts
    layouts = []
    
    # Hexagonal layouts with different aspect ratios of the grid
    for n_cols in range(3, 8):
        n_rows = max(1, int(np.ceil(N / n_cols)))
        if n_cols * n_rows >= N:
            xy, r = _hexagonal_layout(w, h, n_cols, n_rows)
            layouts.append((xy, r))
    
    # Grid layouts
    for n_cols in range(3, 8):
        n_rows = max(1, int(np.ceil(N / n_cols)))
        if n_cols * n_rows >= N:
            xy, r = _grid_layout(w, h, n_cols, n_rows)
            layouts.append((xy, r))
    
    # Random layouts
    for _ in range(5):
        xy, r = _random_layout(w, h)
        layouts.append((xy, r))
    
    # Optimize each layout
    for xy, r in layouts:
        if time.time() - start > time_budget:
            break
        
        time_left = time_budget - (time.time() - start)
        if time_left < 2.0:
            break
        
        xy_opt, r_opt, score = _repair_and_optimize(w, h, xy, r, time_budget=min(time_left * 0.6, 8.0))
        
        if score > best_score:
            best_score = score
            best_xy = xy_opt.copy()
            best_r = r_opt.copy()
        
        # Polish the best result
        if time.time() - start < time_budget - 3.0:
            time_left = time_budget - (time.time() - start)
            if time_left > 3.0:
                xy_pol, r_pol, score_pol = _polish(w, h, best_xy, best_r, time_budget=min(time_left - 3.0, 10.0))
                if score_pol > best_score:
                    best_score = score_pol
                    best_xy = xy_pol.copy()
                    best_r = r_pol.copy()
    
    if best_xy is None:
        # Fallback
        xy, r = _hexagonal_layout(w, h, 5, 5)
        best_xy = xy
        best_r = r
        best_score = _score(w, h, xy, r)
    
    return w, h, best_xy, best_r, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2 (perimeter = 4)
    # aspect = w/h, so w = 2*aspect/(aspect+1), h = 2/(aspect+1)
    aspect_ratios = [1.0, 0.7, 1.4, 0.8, 1.25, 0.6, 1.6, 0.9, 1.1, 0.75, 1.333, 0.667]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (ar + 1.0)
        h = 2.0 / (ar + 1.0)
        
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        time_budget = min(time_left * 0.7, 12.0)
        w_opt, h_opt, xy_opt, r_opt, score = _pack_for_aspect(w, h, time_budget=time_budget)
        
        if score > best_score:
            best_score = score
            best_result = (w_opt, h_opt, xy_opt.copy(), r_opt.copy())
    
    # Final polish on the best result
    if best_result is not None and time.time() < deadline - 5.0:
        w, h, xy, r = best_result
        time_left = deadline - time.time() - 5.0
        if time_left > 3.0:
            xy_pol, r_pol, score_pol = _polish(w, h, xy, r, time_budget=min(time_left, 15.0))
            if score_pol > best_score:
                best_score = score_pol
                best_result = (w, h, xy_pol.copy(), r_pol.copy())
    
    if best_result is None:
        # Fallback: simple grid
        w, h = 1.0, 1.0
        xy, r = _grid_layout(w, h, 5, 5)
        best_result = (w, h, xy, r)
    
    w, h, xy, r = best_result
    
    # Final validity check and repair if needed
    if not _check_validity(w, h, xy, r, tol=1e-6):
        # Shrink radii slightly
        for _ in range(10):
            r *= 0.95
            if _check_validity(w, h, xy, r, tol=1e-6):
                break
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = xy[i, 0]
        result[i, 1] = xy[i, 1]
        result[i, 2] = r[i]
    
    return result
