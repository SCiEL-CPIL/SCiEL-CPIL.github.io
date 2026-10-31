"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # We'll use a gradient-based optimization approach with a penalty for overlaps
    # and constraints for staying in the unit square.
    
    # Initial guess: random positions and radii
    def random_init(seed=42):
        rng = np.random.RandomState(seed)
        # Start with small radii to avoid immediate overlaps
        radii = rng.uniform(0.02, 0.08, size=N)
        x = rng.uniform(0.1, 0.9, size=N)
        y = rng.uniform(0.1, 0.9, size=N)
        return x, y, radii
    
    def compute_loss(x, y, radii):
        """
        Compute loss: negative sum of radii (to maximize) + penalties for overlaps and boundary violations.
        """
        loss = -np.sum(radii)
        
        # Overlap penalty: for each pair, if distance < r_i + r_j, penalize
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    loss += 10.0 * overlap * overlap
        
        # Boundary penalty: circles must stay in [0,1]^2
        for i in range(N):
            if x[i] < radii[i]:
                loss += 10.0 * (radii[i] - x[i]) ** 2
            if x[i] > 1.0 - radii[i]:
                loss += 10.0 * (x[i] - (1.0 - radii[i])) ** 2
            if y[i] < radii[i]:
                loss += 10.0 * (radii[i] - y[i]) ** 2
            if y[i] > 1.0 - radii[i]:
                loss += 10.0 * (y[i] - (1.0 - radii[i])) ** 2
        
        return loss
    
    def check_valid(x, y, radii, tol=1e-9):
        """Check if configuration is valid: no overlaps, all in unit square."""
        for i in range(N):
            if x[i] < -tol or x[i] > 1.0 + tol:
                return False
            if y[i] < -tol or y[i] > 1.0 + tol:
                return False
            if radii[i] < -tol:
                return False
            # Check boundaries
            if x[i] - radii[i] < -tol or x[i] + radii[i] > 1.0 + tol:
                return False
            if y[i] - radii[i] < -tol or y[i] + radii[i] > 1.0 + tol:
                return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def optimize(x0, y0, r0, max_iter=5000, lr=0.001):
        """Simple gradient descent with numerical gradients."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        best_loss = compute_loss(x, y, r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute numerical gradients
            eps = 1e-5
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            grad_r = np.zeros(N)
            
            # Gradient for x
            for i in range(N):
                x_plus = x.copy()
                x_plus[i] += eps
                loss_plus = compute_loss(x_plus, y, r)
                loss_minus = compute_loss(x - 2*eps*np.eye(N)[i,:], y, r) if False else compute_loss(x.copy() - 2*eps*np.eye(N)[i,:], y, r)
                # Simpler: just perturb one at a time
                x_minus = x.copy()
                x_minus[i] -= eps
                loss_minus = compute_loss(x_minus, y, r)
                grad_x[i] = (loss_plus - loss_minus) / (2*eps)
            
            # Gradient for y
            for i in range(N):
                y_plus = y.copy()
                y_plus[i] += eps
                loss_plus = compute_loss(x, y_plus, r)
                y_minus = y.copy()
                y_minus[i] -= eps
                loss_minus = compute_loss(x, y_minus, r)
                grad_y[i] = (loss_plus - loss_minus) / (2*eps)
            
            # Gradient for r
            for i in range(N):
                r_plus = r.copy()
                r_plus[i] += eps
                loss_plus = compute_loss(x, y, r_plus)
                r_minus = r.copy()
                r_minus[i] -= eps
                loss_minus = compute_loss(x, y, r_minus)
                grad_r[i] = (loss_plus - loss_minus) / (2*eps)
            
            # Update with adaptive learning rate
            lr_current = lr / (1.0 + it * 0.01)
            x -= lr_current * grad_x
            y -= lr_current * grad_y
            r -= lr_current * grad_r
            
            # Project to valid range
            r = np.maximum(r, 0.001)
            r = np.minimum(r, 0.4)
            x = np.maximum(x, 0.0)
            x = np.minimum(x, 1.0)
            y = np.maximum(y, 0.0)
            y = np.minimum(y, 1.0)
            
            loss = compute_loss(x, y, r)
            if loss < best_loss:
                best_loss = loss
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        return best_x, best_y, best_r
    
    # Try multiple random initializations
    best_result = None
    best_score = -np.inf
    
    for seed in range(5):
        if time.time() > deadline:
            break
        
        x, y, r = random_init(seed)
        x_opt, y_opt, r_opt = optimize(x, y, r, max_iter=2000, lr=0.005)
        
        if check_valid(x_opt, y_opt, r_opt):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_opt, y_opt, r_opt])
    
    # If no valid result from optimization, try a simple grid-based approach
    if best_result is None:
        # Fall back to a better grid
        # Try different grid arrangements
        best_grid_score = -np.inf
        best_grid = None
        
        for cols in range(4, 7):
            rows = int(np.ceil(N / cols))
            if rows * cols < N:
                continue
            
            dx = 1.0 / cols
            dy = 1.0 / rows
            
            # Max radius for each cell considering neighbors
            radii = []
            xs = []
            ys = []
            
            for i in range(rows):
                for j in range(cols):
                    if len(radii) >= N:
                        break
                    cx = (j + 0.5) * dx
                    cy = (i + 0.5) * dy
                    
                    # Distance to boundaries
                    dist_left = cx
                    dist_right = 1.0 - cx
                    dist_bottom = cy
                    dist_top = 1.0 - cy
                    max_r = min(dist_left, dist_right, dist_bottom, dist_top)
                    
                    # Also consider neighbor distances
                    # For simplicity, use half the cell size
                    max_r = min(max_r, 0.5 * dx, 0.5 * dy)
                    
                    xs.append(cx)
                    ys.append(cy)
                    radii.append(max_r * 0.95)  # Small margin
            
            if len(radii) == N:
                grid = np.column_stack([np.array(xs), np.array(ys), np.array(radii)])
                if check_valid(grid[:,0], grid[:,1], grid[:,2]):
                    score = np.sum(radii)
                    if score > best_grid_score:
                        best_grid_score = score
                        best_grid = grid
        
        if best_grid is not None:
            return best_grid
        else:
            # Last resort: very small circles
            x = np.linspace(0.1, 0.9, N)
            y = np.full(N, 0.5)
            r = np.full(N, 0.01)
            return np.column_stack([x, y, r])
    
    return best_result
# EVOLVE-BLOCK-END
