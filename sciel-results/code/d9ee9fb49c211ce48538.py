import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = -1.0
    best_result = None
    
    # Aspects to try: w/h. Perimeter 4 => w+h=2.
    # w = 2a/(1+a), h = 2/(1+a)
    # Square (1.0) is often optimal for N ~ 20-25 due to symmetry, 
    # but slight deviations can help escape local minima.
    aspects = [1.0, 0.9, 1.1, 0.8, 1.25]
    
    def get_dimensions(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def make_hexagonal_init(w, h, n_circles):
        """
        Generate a hexagonal lattice packing that fits inside w x h.
        This provides a much better starting point than random or simple grid.
        """
        # Estimate radius for hex packing
        # Area efficiency of hex is pi/sqrt(12) ~ 0.9069
        # N * pi * r^2 <= w*h * 0.9
        r_est = np.sqrt(w * h * 0.9 / (n_circles * np.pi))
        
        # Hex spacing
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        
        centers = []
        row = 0
        while len(centers) < n_circles:
            x_offset = 0 if row % 2 == 0 else r_est
            col = 0
            while len(centers) < n_circles:
                x = x_offset + col * dx
                y = row * dy
                
                # Check if inside bounds (with small margin)
                if x >= r_est and x <= w - r_est and y >= r_est and y <= h - r_est:
                    centers.append([x, y])
                    if len(centers) >= n_circles:
                        break
                col += 1
            row += 1
            
            # Safety break if we are going way out of bounds
            if y > h + r_est:
                break
        
        # If we didn't get enough, fill the rest randomly or shrink
        if len(centers) < n_circles:
            # This shouldn't happen with reasonable r_est, but just in case
            # Add random points in valid area
            for _ in range(n_circles - len(centers)):
                cx = random.uniform(r_est, w - r_est)
                cy = random.uniform(r_est, h - r_est)
                centers.append([cx, cy])
        
        centers = np.array(centers[:n_circles])
        radii = np.full(n_circles, r_est * 0.95) # Start slightly smaller to allow growth
        
        return centers, radii

    def make_grid_init(w, h, n_circles):
        """Fallback grid initialization."""
        side = int(np.ceil(np.sqrt(n_circles)))
        centers = np.zeros((n_circles, 2))
        for i in range(n_circles):
            row = i // side
            col = i % side
            centers[i, 0] = (col + 0.5) * (w / side)
            centers[i, 1] = (row + 0.5) * (h / side)
        
        # Estimate radius
        # For grid, max r is min(w/side, h/side) / 2
        r_est = min(w / side, h / side) * 0.5 * 0.95
        radii = np.full(n_circles, r_est)
        return centers, radii

    def check_validity(circles, w, h):
        """Check if circles are disjoint and inside rectangle."""
        xy = circles[:, :2]
        r = circles[:, 2]
        
        # Boundary
        if np.any(xy[:, 0] < r - 1e-5) or np.any(xy[:, 0] > w - r + 1e-5):
            return False
        if np.any(xy[:, 1] < r - 1e-5) or np.any(xy[:, 1] > h - r + 1e-5):
            return False
            
        # Pairwise
        # Use broadcasting for small N
        dists = np.sqrt(np.sum((xy[:, np.newaxis] - xy[np.newaxis, :])**2, axis=-1))
        r_sums = r[:, np.newaxis] + r[np.newaxis, :]
        # Check off-diagonal
        mask = ~np.eye(N, dtype=bool)
        if np.any(dists[mask] < r_sums[mask] - 1e-5):
            return False
        return True

    def objective_and_constraints(x, w, h):
        """
        x: vector of length 3*N: [x_0, y_0, r_0, ..., x_N-1, y_N-1, r_N-1]
        Objective: Minimize -sum(r)
        Constraints:
        1. x_i - r_i >= 0  (Left boundary)
        2. w - x_i - r_i >= 0 (Right boundary)
        3. y_i - r_i >= 0  (Bottom boundary)
        4. h - y_i - r_i >= 0 (Top boundary)
        5. dist(i,j) - r_i - r_j >= 0 (Non-overlap)
        """
        centers = x[0:2*N].reshape(-1, 2)
        radii = x[2*N:3*N]
        
        # Objective: maximize sum of radii -> minimize -sum
        obj_val = -np.sum(radii)
        
        # Boundary constraints (linear)
        # g_i >= 0
        g_bounds = []
        for i in range(N):
            g_bounds.append(centers[i, 0] - radii[i])
            g_bounds.append(w - centers[i, 0] - radii[i])
            g_bounds.append(centers[i, 1] - radii[i])
            g_bounds.append(h - centers[i, 1] - radii[i])
        
        # Pairwise constraints (non-linear)
        g_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                g_pairs.append(dist - radii[i] - radii[j])
        
        return obj_val, g_bounds, g_pairs

    # Main Loop
    random.seed(42)
    np.random.seed(42)
    
    # We will try multiple aspects. For each, we do a few restarts.
    # SLSQP can be slow, so we limit iterations and restarts.
    
    restarts_per_aspect = 3
    
    for aspect in aspects:
        w, h = get_dimensions(aspect)
        
        for restart in range(restarts_per_aspect):
            if time.time() > deadline - 10:  # Leave 10s for final validation/return
                break
                
            # Generate initial guess
            if restart == 0:
                centers, radii = make_hexagonal_init(w, h, N)
            elif restart == 1:
                centers, radii = make_grid_init(w, h, N)
            else:
                # Perturbed hex
                centers, radii = make_hexagonal_init(w, h, N)
                centers += np.random.normal(0, 0.01, centers.shape)
                # Ensure within bounds roughly
                centers[:, 0] = np.clip(centers[:, 0], radii, w - radii)
                centers[:, 1] = np.clip(centers[:, 1], radii, h - radii)
            
            # Flatten for SLSQP
            x0 = np.concatenate([centers.flatten(), radii.flatten()])
            
            # Bounds
            # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
            r_max = min(w, h) / 2.0
            bounds = []
            for i in range(N):
                bounds.append((0.0, w))  # x_i
                bounds.append((0.0, h))  # y_i
                bounds.append((0.0, r_max))  # r_i
            
            # Constraints function for SLSQP
            # SLSQP expects constraints as {'type': 'ineq', 'fun': lambda x: ...}
            # where fun(x) >= 0
            
            def constraint_func(x):
                _, g_bounds, g_pairs = objective_and_constraints(x, w, h)
                return np.concatenate([g_bounds, g_pairs])
                
            # We also need to handle the objective separately for minimize
            def obj_func(x):
                obj_val, _, _ = objective_and_constraints(x, w, h)
                return obj_val
            
            constraints = ({'type': 'ineq', 'fun': constraint_func})
            
            try:
                res = minimize(
                    obj_func, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.success or res.fun < -1e-6: # res.fun is -sum(r), so negative means positive sum
                    x_opt = res.x
                    circles = np.zeros((N, 3))
                    circles[:, 0] = x_opt[0:2*N:2] # Wait, x is interleaved? No, x is [x0,y0,r0, x1,y1,r1...]
                    # My x0 construction: centers.flatten() is [x0,y0, x1,y1...], radii is [r0, r1...]
                    # So x_opt[0:2*N] is centers, x_opt[2*N:3*N] is radii
                    centers_opt = x_opt[0:2*N].reshape(-1, 2)
                    radii_opt = x_opt[2*N:3*N]
                    
                    circles[:, 0] = centers_opt[:, 0]
                    circles[:, 1] = centers_opt[:, 1]
                    circles[:, 2] = radii_opt
                    
                    # Strict validity check
                    if check_validity(circles, w, h):
                        score = np.sum(radii_opt)
                        if score > best_score:
                            best_score = score
                            best_result = circles.copy()
                            
            except Exception as e:
                # SLSQP can fail if initial guess is invalid or Jacobian issues
                pass

    if best_result is None:
        # Fallback: Grid of tiny circles
        w, h = 1.0, 1.0
        side = 5
        res = np.zeros((N, 3))
        r = 0.08
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    return best_result
