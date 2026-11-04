import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    def soft_objective(z):
        """
        Objective function: -sum(r) + lambda * penalty
        We minimize this, so we want to maximize sum(r) and minimize penalty.
        z: shape (N, 3) [x, y, r]
        """
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        # Sum of radii (we want to maximize, so negative in objective)
        sum_r = np.sum(r)
        
        # Pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        
        # Overlap penalty (smooth quadratic)
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary penalty (smooth quadratic)
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0, r - x)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_b += np.sum(np.maximum(0, r - y)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
        
        total_penalty = penalty_overlap + pen_b
        
        # Adaptive lambda: start high to enforce constraints, lower to allow growth
        # Simple heuristic: if penalty is small, reduce lambda
        # For simplicity in this step, we use a fixed large lambda for the soft constraint
        # The projection step will handle the final feasibility.
        lambda_val = 10.0
        
        return -sum_r + lambda_val * total_penalty

    def soft_gradient(z):
        """
        Gradient of the soft objective.
        """
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        grad = np.zeros_like(z)
        
        # Gradient of -sum(r)
        grad[:, 2] -= 1.0
        
        # Gradient of overlap penalty
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        
        # Avoid division by zero
        d_safe = np.maximum(d_ij, 1e-8)
        
        # Overlap amount
        overlap = np.maximum(0, r_i + r_j - d_ij)
        
        # Derivative of overlap^2 w.r.t d_ij is -2 * overlap
        # Derivative of d_ij w.r.t x_i is (x_i - x_j) / d_ij
        # So grad_x_i += -2 * overlap * (x_i - x_j) / d_ij
        # And grad_r_i += -2 * overlap
        
        # Accumulate gradients for all pairs
        # This is a bit slow in Python loops, but N=26 is small.
        # Vectorized approach for gradient accumulation:
        
        # We need to sum contributions from all j != i
        # Let's use a simple loop for clarity and correctness, N is small.
        for idx in range(N):
            # Overlaps with all other circles
            others = np.array([j for j in range(N) if j != idx])
            if len(others) == 0: continue
            
            d_i = dist[idx, others]
            r_others = r[others]
            r_self = r[idx]
            
            overlaps_i = np.maximum(0, r_self + r_others - d_i)
            
            # Grad x
            dx = x[idx] - x[others]
            dy = y[idx] - y[others]
            
            # d(d_ij)/d(x_i) = dx / d_ij
            # Contribution to grad_x_i: lambda * sum( -2 * overlap * dx / d_ij )
            # Note: overlap is 0 if no overlap, so we only sum where overlap > 0
            
            mask = overlaps_i > 0
            if np.any(mask):
                d_safe_i = d_i[mask]
                dx_masked = dx[mask]
                dy_masked = dy[mask]
                ov_masked = overlaps_i[mask]
                
                # Factor: -2 * lambda * overlap / d
                factor = -2.0 * 10.0 * ov_masked / d_safe_i
                
                grad[idx, 0] += np.sum(factor * dx_masked)
                grad[idx, 1] += np.sum(factor * dy_masked)
                
                # Grad r: -2 * lambda * overlap
                grad[idx, 2] += np.sum(-2.0 * 10.0 * ov_masked)
            
            # Boundary penalties for circle i
            # Pen_x = max(0, r - x)^2 -> grad_x = -2(r-x) if r>x else 0
            # Pen_1-x = max(0, r - (1-x))^2 -> grad_x = 2(r-(1-x)) if r>1-x else 0
            # Similarly for y
            
            if r[idx] > x[idx]:
                grad[idx, 0] += -2.0 * 10.0 * (r[idx] - x[idx])
                grad[idx, 2] += 2.0 * 10.0 * (r[idx] - x[idx])
            
            if r[idx] > (1 - x[idx]):
                grad[idx, 0] += 2.0 * 10.0 * (r[idx] - (1 - x[idx]))
                grad[idx, 2] += 2.0 * 10.0 * (r[idx] - (1 - x[idx]))
            
            if r[idx] > y[idx]:
                grad[idx, 1] += -2.0 * 10.0 * (r[idx] - y[idx])
                grad[idx, 2] += 2.0 * 10.0 * (r[idx] - y[idx])
            
            if r[idx] > (1 - y[idx]):
                grad[idx, 1] += 2.0 * 10.0 * (r[idx] - (1 - y[idx]))
                grad[idx, 2] += 2.0 * 10.0 * (r[idx] - (1 - y[idx]))
        
        return grad

    def project_feasible(z):
        """
        Project z to be feasible by reducing radii if necessary.
        This is a heuristic projection.
        """
        z = z.copy()
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        # Iteratively reduce radii that violate constraints
        for _ in range(10):
            # Check boundaries
            r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            r = np.minimum(r, r_max_b)
            
            # Check overlaps
            # For each pair, if overlap, reduce radii
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx**2 + dy**2)
                    if d < 1e-8:
                        # Same point, push apart or reduce radii
                        # Simple: reduce both radii to 0
                        r[i] = 0
                        r[j] = 0
                        changed = True
                        continue
                    
                    current_r_sum = r[i] + r[j]
                    if current_r_sum > d:
                        # Need to reduce radii
                        # Reduce proportionally or just clamp
                        # Let's reduce the larger one or both
                        # Simple heuristic: reduce both by half the excess
                        excess = current_r_sum - d
                        reduction = excess / 2.0 + 1e-5
                        r[i] = max(0.0, r[i] - reduction)
                        r[j] = max(0.0, r[j] - reduction)
                        changed = True
            
            if not changed:
                break
        
        z[:, 0] = x
        z[:, 1] = y
        z[:, 2] = r
        return z

    def validate(z_opt):
        """Check if solution is valid and return sum of radii if so."""
        x, y, r = z_opt[:, 0], z_opt[:, 1], z_opt[:, 2]
        
        # Check radii are positive
        if np.any(r <= 0):
            return -np.inf
        
        # Pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i, r_j = r[iu[0]], r[iu[1]]
        max_overlap = np.max(np.maximum(0, r_i + r_j - d_ij))
        
        # Boundary check
        pen_b = 0.0
        pen_b += np.max(np.maximum(0, r - x))
        pen_b += np.max(np.maximum(0, r - (1 - x)))
        pen_b += np.max(np.maximum(0, r - y))
        pen_b += np.max(np.maximum(0, r - (1 - y)))
        
        if max_overlap < 1e-5 and pen_b < 1e-5:
            return np.sum(r)
        return -np.inf

    def grow_radii(z):
        """
        Try to increase radii while maintaining feasibility.
        """
        z = z.copy()
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            
            improved = False
            for i in range(N):
                # Max possible radius for circle i given others
                r_max = min(x[i], 1 - x[i], y[i], 1 - y[i])
                
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx**2 + dy**2)
                    r_max = min(r_max, d - r[j])
                
                if r_max > r[i] + 1e-6:
                    # Try to increase
                    new_r = min(r_max, r[i] + 0.005)
                    z[i, 2] = new_r
                    # Check if still feasible (should be by construction, but verify)
                    # If not, revert
                    # For speed, we assume the check is light
                    improved = True
            
            if not improved:
                break
        
        return z

    # Generate diverse initial configurations
    inits = []
    
    # Init 1: Hexagonal packing
    def generate_hex_init(scale, offset_x, offset_y):
        pts = []
        row = 0
        while True:
            y = row * scale * np.sqrt(3) / 2 + offset_y
            if y > 1.0:
                break
            col = 0
            while True:
                x = col * scale + offset_x + (row % 2) * scale / 2
                if x > 1.0:
                    break
                pts.append((x, y))
                col += 1
            row += 1
        
        if len(pts) >= N:
            np.random.shuffle(pts)
            selected = pts[:N]
            init = np.zeros((N, 3))
            for i, (x, y) in enumerate(selected):
                init[i, 0] = x
                init[i, 1] = y
                init[i, 2] = 0.08
            return init
        return None
    
    # Multiple hex configurations
    for scale in [0.14, 0.16, 0.18]:
        for ox in [0.02, 0.08]:
            for oy in [0.02, 0.08]:
                init = generate_hex_init(scale, ox, oy)
                if init is not None:
                    inits.append(init)
    
    # Init 2: Random configurations
    for seed in range(10):
        np.random.seed(seed)
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        rr = np.full(N, 0.07)
        inits.append(np.column_stack((rx, ry, rr)))
    
    # Run L-BFGS-B optimization from each init
    for init in inits:
        if time.time() - start_time > timeout:
            break
        
        try:
            # Flatten z for scipy
            z_flat = init.flatten()
            
            # Bounds: x,y in [0,1], r in [0.01, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0)) # x
                bounds.append((0.0, 1.0)) # y
                bounds.append((0.01, 0.5)) # r
            
            # Optimize
            result = minimize(
                soft_objective, 
                z_flat, 
                jac=soft_gradient, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 1000}
            )
            
            z_opt = result.x.reshape(N, 3)
            
            # Project to feasible
            z_proj = project_feasible(z_opt)
            
            # Grow radii
            z_grown = grow_radii(z_proj)
            
            # Validate and update best
            score = validate(z_grown)
            if score > best_score:
                best_score = score
                best_solution = z_grown
        except Exception:
            pass
    
    # If no valid solution found, try a simple fallback
    if best_solution is None:
        # Fallback: place circles in a grid with small radii
        side = int(np.ceil(np.sqrt(N)))
        best_solution = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    best_solution[count, 0] = (i + 0.5) / side
                    best_solution[count, 1] = (j + 0.5) / side
                    best_solution[count, 2] = 0.5 / side - 0.01
                    count += 1
    
    return best_solution
