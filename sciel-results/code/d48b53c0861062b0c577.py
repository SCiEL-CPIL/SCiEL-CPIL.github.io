"""Circle packing in perimeter-4 rectangle: Physics-based soft-body optimization."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None

    def _is_valid(cx, cy, r, W, H, tol=1e-9):
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

    def _physics_solve(w, h, x_init, y_init, r_init, max_iters=2000, time_limit=None):
        """
        Iteratively adjust positions and radii using repulsion forces.
        """
        x = x_init.copy().astype(float)
        y = y_init.copy().astype(float)
        r = r_init.copy().astype(float)
        
        # Parameters
        repulsion_strength = 1.0
        wall_repulsion_strength = 1.0
        damping = 0.9
        expansion_rate = 0.002
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force = repulsion_strength * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    
                    # Also apply a gentle repulsion even if not overlapping to separate them
                    # This helps in finding dense packings
                    if dist < min_dist * 1.5:
                        gentle_force = 0.1 * (min_dist * 1.5 - dist)
                        fx[i] += gentle_force * dx / dist
                        fy[i] += gentle_force * dy / dist
                        fx[j] -= gentle_force * dx / dist
                        fy[j] -= gentle_force * dy / dist

            # Wall repulsion
            for i in range(N):
                # Left
                if x[i] - r[i] < 0:
                    fx[i] += wall_repulsion_strength * (r[i] - x[i])
                # Right
                if x[i] + r[i] > w:
                    fx[i] -= wall_repulsion_strength * (x[i] + r[i] - w)
                # Bottom
                if y[i] - r[i] < 0:
                    fy[i] += wall_repulsion_strength * (r[i] - y[i])
                # Top
                if y[i] + r[i] > h:
                    fy[i] -= wall_repulsion_strength * (y[i] + r[i] - h)

            # Update positions
            x += fx * damping
            y += fy * damping
            
            # Clamp positions to bounds (soft constraint)
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)

            # Expand radii slightly if no strong repulsion
            # We expand all radii by a small amount, relying on repulsion to push them apart
            # If they can't be pushed apart, they will eventually be clamped or overlap,
            # but the repulsion force will grow to counteract this.
            r += expansion_rate
            r = np.maximum(r, 0.01)

        return x, y, r

    def _greedy_expand(x, y, r, w, h, time_budget):
        """Iteratively expand each circle to its maximum possible radius."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        improved = True
        rounds = 0
        while improved and rounds < 50:
            if time.time() > time_budget:
                break
            improved = False
            rounds += 1
            for i in range(N):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
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

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65]
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
            
        # Initialize grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4

        # Run physics simulation
        time_limit = deadline - 5.0
        x_sol, y_sol, r_sol = _physics_solve(w0, h0, x, y, r, max_iters=1000, time_limit=time_limit)
        
        # Check validity and expand
        if _is_valid(x_sol, y_sol, r_sol, w0, h0):
            x_exp, y_exp, r_exp = _greedy_expand(x_sol, y_sol, r_sol, w0, h0, deadline - 2.0)
            if _is_valid(x_exp, y_exp, r_exp, w0, h0):
                val = float(np.sum(r_exp))
                if val > best_value:
                    best_value = val
                    best_result = (x_exp, y_exp, r_exp, w0, h0)
        
        # Also check the raw physics solution in case expansion failed or reduced value (unlikely)
        if _is_valid(x_sol, y_sol, r_sol, w0, h0):
             val = float(np.sum(r_sol))
             if val > best_value:
                 best_value = val
                 best_result = (x_sol, y_sol, r_sol, w0, h0)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
