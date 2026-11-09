"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    def is_valid(positions, radii, w, h, tol=1e-7):
        for i in range(N):
            if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
                return False
            if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def compute_score(positions, radii, w, h):
        if not is_valid(positions, radii, w, h):
            return -1.0
        return np.sum(radii)
    
    def hexagonal_positions(w, h, cols, rows):
        """Generate hexagonal packing positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += dx * 0.25  # Offset odd rows
                positions.append([x, y])
        return np.array(positions[:N])
    
    def grid_positions(w, h, cols, rows):
        """Generate grid positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                positions.append([x, y])
        return np.array(positions[:N])
    
    def initial_radii(positions, w, h, cols, rows):
        """Compute initial radii based on spacing and boundaries."""
        dx = w / cols
        dy = h / rows
        r_init = 0.5 * min(dx, dy) * 0.9
        radii = []
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            radii.append(min(r_init, r_max * 0.95))
        return np.array(radii)
    
    def relax(positions, radii, w, h, max_iter=300):
        """Force-based relaxation to improve packing."""
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Check validity
            if not is_valid(positions, radii, w, h):
                # Push circles back into bounds
                for i in range(N):
                    positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                    positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
                continue
            
            # Compute forces
            forces = np.zeros_like(positions)
            dradii = np.zeros(N)
            
            # Boundary forces
            for i in range(N):
                if positions[i, 0] - radii[i] < 0.01:
                    forces[i, 0] += 5.0 * (0.01 - (positions[i, 0] - radii[i]))
                if positions[i, 0] + radii[i] > w - 0.01:
                    forces[i, 0] -= 5.0 * ((positions[i, 0] + radii[i]) - (w - 0.01))
                if positions[i, 1] - radii[i] < 0.01:
                    forces[i, 1] += 5.0 * (0.01 - (positions[i, 1] - radii[i]))
                if positions[i, 1] + radii[i] > h - 0.01:
                    forces[i, 1] -= 5.0 * ((positions[i, 1] + radii[i]) - (h - 0.01))
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                    elif dist < 1e-10:
                        forces[i, 0] += np.random.randn() * 0.005
                        forces[i, 1] += np.random.randn() * 0.005
                        forces[j, 0] -= np.random.randn() * 0.005
                        forces[j, 1] -= np.random.randn() * 0.005
            
            # Try to expand radii
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - radii[j]) / 1.0)
                r_max = max(r_max, 0.001)
                dradii[i] += (r_max - radii[i]) * 0.1
            
            # Apply updates
            lr = 0.03
            positions += forces * lr
            radii += dradii * lr
            
            # Clamp
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        return positions, radii
    
    def optimize_with_scipy(positions, radii, w, h):
        """Use scipy optimizer to refine the packing."""
        n = N
        # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        def objective(x):
            pos = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
            rad = np.array([x[3*i+2] for i in range(n)])
            
            # Penalty for violations
            penalty = 0.0
            for i in range(n):
                if pos[i, 0] - rad[i] < 0:
                    penalty += 1000 * (pos[i, 0] - rad[i]) ** 2
                if pos[i, 0] + rad[i] > w:
                    penalty += 1000 * (pos[i, 0] + rad[i] - w) ** 2
                if pos[i, 1] - rad[i] < 0:
                    penalty += 1000 * (pos[i, 1] - rad[i]) ** 2
                if pos[i, 1] + rad[i] > h:
                    penalty += 1000 * (pos[i, 1] + rad[i] - h) ** 2
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist:
                        penalty += 1000 * (min_dist - dist) ** 2
            
            # Maximize sum of radii => minimize negative sum
            return -np.sum(rad) + penalty
        
        # Bounds: positions within [0, w]x[0, h], radii positive
        bounds = []
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0.001, min(w, h)/2))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < 0:
                pos = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
                rad = np.array([result.x[3*i+2] for i in range(n)])
                
                # Check validity
                if is_valid(pos, rad, w, h):
                    return pos, rad
        except:
            pass
        
        return None, None
    
    # Try different aspect ratios and grid configurations
    configs = [
        # (aspect_ratio, cols, rows, use_hex)
        (1.0, 5, 5, False),
        (1.0, 5, 5, True),
        (1.2, 6, 4, False),
        (1.2, 6, 4, True),
        (0.8, 4, 6, False),
        (0.8, 4, 6, True),
        (1.5, 7, 3, False),
        (1.5, 7, 3, True),
        (0.67, 3, 7, False),
        (0.67, 3, 7, True),
        (2.0, 7, 3, False),
        (2.0, 7, 3, True),
        (0.5, 3, 7, False),
        (0.5, 3, 7, True),
        (1.4, 6, 4, False),
        (1.4, 6, 4, True),
        (0.7, 4, 6, False),
        (0.7, 4, 6, True),
        (1.3, 6, 4, False),
        (1.3, 6, 4, True),
        (1.1, 5, 5, False),
        (1.1, 5, 5, True),
        (0.9, 5, 5, False),
        (0.9, 5, 5, True),
        (1.6, 7, 3, False),
        (1.6, 7, 3, True),
        (0.63, 3, 7, False),
        (0.63, 3, 7, True),
    ]
    
    for ar, cols, rows, use_hex in configs:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))
        h = 2.0 - w
        
        # Ensure we have at least N cells
        if cols * rows < N:
            continue
        
        # Generate initial positions
        if use_hex:
            positions = hexagonal_positions(w, h, cols, rows)
        else:
            positions = grid_positions(w, h, cols, rows)
        
        if len(positions) < N:
            continue
        
        positions = positions[:N]
        radii = initial_radii(positions, w, h, cols, rows)
        
        # First pass: force-based relaxation
        positions, radii = relax(positions, radii, w, h, max_iter=200)
        
        score = compute_score(positions, radii, w, h)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()
        
        # Second pass: scipy optimization if time permits
        if time.time() < deadline - 10:
            opt_pos, opt_rad = optimize_with_scipy(positions, radii, w, h)
            if opt_pos is not None:
                opt_score = compute_score(opt_pos, opt_rad, w, h)
                if opt_score > best_score:
                    best_score = opt_score
                    best_result = np.column_stack([opt_pos, opt_rad]).copy()
        
        # Third pass: more relaxation after scipy
        if time.time() < deadline - 5 and best_result is not None:
            # Try to improve the best result further
            pos_copy = best_result[:, :2].copy()
            rad_copy = best_result[:, 2].copy()
            pos_copy, rad_copy = relax(pos_copy, rad_copy, w, h, max_iter=100)
            final_score = compute_score(pos_copy, rad_copy, w, h)
            if final_score > best_score:
                best_score = final_score
                best_result = np.column_stack([pos_copy, rad_copy]).copy()
    
    # If no valid result found, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
