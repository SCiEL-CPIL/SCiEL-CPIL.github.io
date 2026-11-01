"""Circle packing in perimeter-4 rectangle: gradient-free Nelder-Mead + greedy expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _penalized_objective(z, penalty_weight=5000.0):
    """Penalized objective for gradient-free optimization."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3 or np.any(r < 0.001):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary
    lv = np.maximum(r - x, 0.0)
    rv = np.maximum(x + r - w, 0.0)
    bv = np.maximum(r - y, 0.0)
    tv = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(lv*lv) + np.sum(rv*rv) + np.sum(bv*bv) + np.sum(tv*tv))

    # Aspect ratio penalty - encourage near-square for better packing
    ideal_w = 1.0
    obj += 50.0 * (w - ideal_w) ** 2

    return obj


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    orderings = [list(range(N)), list(range(N-1, -1, -1)), 
                 sorted(range(N), key=lambda i: r[i]),
                 sorted(range(N), key=lambda i: -r[i])]
    while improved and rounds < 80:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        # Cycle through different orderings
        order = orderings[rounds % len(orderings)]
        for i in order:
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _local_search_positions(x, y, r, w, h, time_budget):
    """Try small random perturbations of positions to find better packings."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(123)
    best_sum = float(np.sum(r))
    
    # Simulated annealing style: start with larger steps, decrease
    step_scale = 0.02
    for iteration in range(8000):
        if time.time() > time_budget:
            break
        step_scale *= 0.9995  # Slowly decrease step size
        
        # Pick 1-3 circles to nudge simultaneously
        n_move = int(rng.integers(1, 4))
        indices = rng.choice(N, size=n_move, replace=False)
        
        moves = {}
        valid = True
        for i in indices:
            dx = rng.normal(0, step_scale)
            dy = rng.normal(0, step_scale)
            nx = x[i] + dx
            ny = y[i] + dy
            if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                valid = False
                break
            moves[i] = (nx, ny)
        
        if not valid:
            continue
        
        # Check all pairwise constraints with new positions
        for i in indices:
            nx, ny = moves[i]
            for j in range(N):
                if i == j:
                    continue
                if j in moves:
                    jx, jy = moves[j]
                else:
                    jx, jy = x[j], y[j]
                ddx = nx - jx
                ddy = ny - jy
                if ddx*ddx + ddy*ddy < (r[i] + r[j])**2 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Apply moves
            for i in indices:
                x[i], y[i] = moves[i]
            # Try to expand all circles
            x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
            new_sum = float(np.sum(r2))
            if new_sum > best_sum:
                best_sum = new_sum
                r = r2
                x = x2
                y = y2
            else:
                # Revert
                for i in indices:
                    dx, dy = moves[i]
                    x[i] -= dx
                    y[i] -= dy
                r = r.copy()
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_result = None

    def check_validity(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return True, float(np.sum(r))
        return False, 0.0

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid-based initial guesses with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65, 1.2, 0.85]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        # Hexagonal packing initial guess
        for w0 in [1.0, 0.9, 1.1]:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            # Hexagonal arrangement
            rows = int(np.ceil(N / 5))
            for i in range(N):
                row = i // 5
                col = i % 5
                offset = 0.5 if row % 2 == 1 else 0.0
                cx = (col + 0.5 + offset) / 5.0 * w0
                cy = (row + 0.5) / rows * h0
                r0 = min(w0 / 5.0, h0 / rows) * 0.38
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        # Corner-pinned arrangement: large circles in corners
        for w0 in [1.0, 0.95, 1.05]:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            corner_r = min(w0, h0) * 0.18
            # Place 4 large circles in corners
            positions = [
                (corner_r, corner_r),
                (w0 - corner_r, corner_r),
                (corner_r, h0 - corner_r),
                (w0 - corner_r, h0 - corner_r)
            ]
            for i in range(4):
                z[3*i] = positions[i][0]
                z[3*i+1] = positions[i][1]
                z[3*i+2] = corner_r
            # Fill remaining with smaller circles in grid
            remaining = N - 4
            grid_side = int(np.ceil(np.sqrt(remaining)))
            for i in range(remaining):
                row = i // grid_side
                col = i % grid_side
                cx = (col + 0.5) / grid_side * w0
                cy = (row + 0.5) / grid_side * h0
                r0 = min(w0 / grid_side, h0 / grid_side) * 0.35
                z[3*(4+i)] = cx
                z[3*(4+i)+1] = cy
                z[3*(4+i)+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        # Random starts with better distribution
        rng = np.random.default_rng(42)
        for _ in range(20):
            w0 = rng.uniform(0.75, 1.25)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            # Use Poisson disk-like sampling: place circles with minimum spacing
            placed = []
            for i in range(N):
                attempts = 0
                while attempts < 100:
                    r0 = rng.uniform(0.04, 0.09)
                    cx = rng.uniform(r0, w0 - r0)
                    cy = rng.uniform(r0, h0 - r0)
                    # Check if it doesn't overlap with placed circles
                    valid = True
                    for (px, py, pr) in placed:
                        ddx = cx - px
                        ddy = cy - py
                        if ddx*ddx + ddy*ddy < (r0 + pr)**2:
                            valid = False
                            break
                    if valid:
                        placed.append((cx, cy, r0))
                        z[3*i] = cx
                        z[3*i+1] = cy
                        z[3*i+2] = r0
                        break
                    attempts += 1
                if attempts >= 100:
                    # Fallback: just place it
                    r0 = 0.04
                    cx = rng.uniform(r0, w0 - r0)
                    cy = rng.uniform(r0, h0 - r0)
                    z[3*i] = cx
                    z[3*i+1] = cy
                    z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break
        
        # Nelder-Mead optimization (gradient-free)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(4000, max(300, remaining * 50)))
        
        try:
            res = minimize(_penalized_objective, z0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True, 'initial_simplex': None})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try to repair by shrinking radii progressively
                for shrink in [0.95, 0.9, 0.85, 0.8]:
                    z_repair = z_opt.copy()
                    r_idx = np.arange(2, len(z_repair)-1, 3)
                    z_repair[r_idx] *= shrink
                    valid, val = check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair.copy()
                        break
        except Exception:
            continue

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Polish: greedy expansion + local search (multiple rounds)
    if time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        # Multiple rounds of expansion and local search
        for round_num in range(3):
            # Greedy expansion
            x, y, r = _greedy_expand(x, y, r, w, h, deadline - 5)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

            # Local search on positions
            if time.time() < deadline - 5:
                x = best_result[0::3][:N].copy()
                y = best_result[1::3][:N].copy()
                r = best_result[2::3][:N].copy()
                w = best_result[-1]
                h = 2.0 - w
                x, y, r = _local_search_positions(x, y, r, w, h, deadline - 3)
                if _is_valid(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3*N+1)
                        best_result[0::3][:N] = x
                        best_result[1::3][:N] = y
                        best_result[2::3][:N] = r
                        best_result[-1] = w

        # Final greedy expansion
        if time.time() < deadline - 2:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
