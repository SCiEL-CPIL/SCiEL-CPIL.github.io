import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

N = 26

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # We'll use a contact-graph approach where we define a set of "anchor" circles
    # and derive other circle positions from contact constraints.
    # For simplicity and robustness, we'll use a hybrid approach:
    # 1. Start with a good grid/hexagonal layout
    # 2. Use differential evolution on a reduced set of parameters
    # 3. The parameters control the overall scale and perturbation pattern
    
    def compute_packing_from_params(params):
        """
        params: array of shape (n_params,)
        We'll use a structured approach:
        - First 2 params: global scale and offset
        - Remaining params: perturbations for each circle
        """
        n_params = len(params)
        
        # Base layout: 5x5 grid (25 circles) + 1 extra
        side = 5
        base_x = []
        base_y = []
        base_r = []
        
        for i in range(side):
            for j in range(side):
                base_x.append((i + 0.5) / side)
                base_y.append((j + 0.5) / side)
                base_r.append(0.5 / side)
        
        # Add 26th circle in the center
        base_x.append(0.5)
        base_y.append(0.5)
        base_r.append(0.5 / side * 0.8)
        
        base_x = np.array(base_x[:N])
        base_y = np.array(base_y[:N])
        base_r = np.array(base_r[:N])
        
        # Extract parameters
        scale = params[0]
        cx = params[1]
        cy = params[2]
        
        # Perturbations for each circle (3 per circle: dx, dy, dr)
        perturb = params[3:]
        dx = perturb[:N]
        dy = perturb[N:2*N]
        dr = perturb[2*N:3*N]
        
        # Apply transformations
        x = cx + scale * (base_x - 0.5) + dx
        y = cy + scale * (base_y - 0.5) + dy
        r = np.maximum(base_r * scale + dr, 0.01)
        
        # Clip to ensure circles stay in bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Ensure radii don't exceed what's possible given position
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, max_r)
        r = np.maximum(r, 0.01)
        
        # Re-clip positions after adjusting radii
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        return x, y, r
    
    def objective(params):
        """
        Minimize negative sum of radii plus overlap penalty
        """
        if time.time() - start_time > timeout:
            return 1e10
        
        x, y, r = compute_packing_from_params(params)
        
        # Calculate sum of radii
        sum_r = np.sum(r)
        
        # Check for overlaps
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        
        r_sum = np.add.outer(r, r)
        overlaps = np.maximum(0, r_sum - d)
        overlap_pen = np.sum(overlaps**2)
        
        # Check boundary violations (should be minimal after clipping)
        bound_viol = np.maximum(0, r - x) + np.maximum(0, x - (1 - r)) + \
                     np.maximum(0, r - y) + np.maximum(0, y - (1 - r))
        bound_pen = np.sum(bound_viol**2)
        
        total_pen = overlap_pen + bound_pen
        
        # Heavily penalize violations
        lambda_pen = 100.0
        return -sum_r + lambda_pen * total_pen
    
    # Define bounds for differential evolution
    # scale: [0.8, 1.2], cx: [0.4, 0.6], cy: [0.4, 0.6]
    # dx, dy: [-0.1, 0.1], dr: [-0.05, 0.05]
    bounds = [
        (0.8, 1.2),  # scale
        (0.4, 0.6),  # cx
        (0.4, 0.6),  # cy
    ]
    for _ in range(N):
        bounds.append((-0.1, 0.1))  # dx
    for _ in range(N):
        bounds.append((-0.1, 0.1))  # dy
    for _ in range(N):
        bounds.append((-0.05, 0.05))  # dr
    
    n_total_params = 3 + 3 * N
    
    # Run differential evolution
    best_val = np.inf
    best_params = None
    
    try:
        # Use differential evolution with limited iterations
        result = differential_evolution(
            objective, 
            bounds,
            maxiter=200,
            tol=1e-5,
            seed=42,
            polish=True,
            popsize=15,
            update_interval=10,
            time_limit=timeout - 5
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception as e:
        pass
    
    # If DE failed or timed out, try a few random restarts with local optimization
    if best_params is None or time.time() - start_time > timeout - 30:
        for trial in range(5):
            if time.time() - start_time > timeout - 10:
                break
            
            # Random initial parameters
            x0 = np.array([
                np.random.uniform(0.9, 1.1),  # scale
                np.random.uniform(0.45, 0.55),  # cx
                np.random.uniform(0.45, 0.55),  # cy
            ])
            for _ in range(N):
                x0 = np.append(x0, np.random.uniform(-0.05, 0.05))  # dx
            for _ in range(N):
                x0 = np.append(x0, np.random.uniform(-0.05, 0.05))  # dy
            for _ in range(N):
                x0 = np.append(x0, np.random.uniform(-0.02, 0.02))  # dr
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception as e:
                pass
    
    if best_params is None:
        # Fallback: simple grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        return np.array(cells)
    
    x, y, r = compute_packing_from_params(best_params)
    
    # Post-processing: ensure validity
    # Iteratively shrink radii if there are overlaps
    max_iter = 20
    for _ in range(max_iter):
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        overlaps = np.maximum(0, r_sum - d)
        max_overlap = np.max(overlaps)
        
        # Check boundaries
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        if max_overlap < 1e-8 and min_x >= 0 and max_x <= 1 and min_y >= 0 and max_y <= 1:
            break
        
        # Shrink all radii slightly
        r = r * 0.995
        
        # Re-clip
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        r = np.maximum(r, 0.005)
    
    # Final validity check and adjustment
    x_diff = np.subtract.outer(x, x)
    y_diff = np.subtract.outer(y, y)
    d_sq = x_diff**2 + y_diff**2
    np.fill_diagonal(d_sq, np.inf)
    d = np.sqrt(d_sq)
    r_sum = np.add.outer(r, r)
    overlaps = np.maximum(0, r_sum - d)
    
    if np.max(overlaps) > 1e-8:
        # More aggressive shrinking
        for _ in range(50):
            r = r * 0.99
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            r = np.maximum(r, 0.001)
            
            x_diff = np.subtract.outer(x, x)
            y_diff = np.subtract.outer(y, y)
            d_sq = x_diff**2 + y_diff**2
            np.fill_diagonal(d_sq, np.inf)
            d = np.sqrt(d_sq)
            r_sum = np.add.outer(r, r)
            overlaps = np.maximum(0, r_sum - d)
            
            if np.max(overlaps) < 1e-10:
                break
    
    return np.column_stack((x, y, r))
