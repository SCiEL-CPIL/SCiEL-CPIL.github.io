"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s limit
    
    # Strategy: Use hexagonal lattice arrangement and optimize
    # Try multiple initial configurations and optimize
    
    best_result = None
    best_score = 0.0
    
    # Helper to evaluate a configuration
    def evaluate_config(positions, radii, w, h):
        """Check validity and compute score (sum of radii)"""
        n = len(positions)
        # Check boundaries
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            if x < r - 1e-9 or x > w - r + 1e-9 or y < r - 1e-9 or y > h - r + 1e-9:
                return -1.0
        # Check non-overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return -1.0
        return np.sum(radii)
    
    # Generate hexagonal lattice configurations
    def make_hex_config(rows_pattern, aspect_ratio):
        """Create hexagonal packing with given row pattern and aspect ratio"""
        w = aspect_ratio
        h = 2.0 - w
        
        # Calculate spacing from the pattern
        # For hexagonal packing, horizontal spacing = d, vertical spacing = d*sqrt(3)/2
        # Try to fit the pattern
        
        positions = []
        radii = []
        
        # Estimate radius from fitting
        # For a hexagonal pattern with 'rows' rows and varying counts
        n_rows = len(rows_pattern)
        max_per_row = max(rows_pattern)
        
        # Horizontal extent: (max_per_row - 1) * d + 2*r = (max_per_row-1)*d + d = max_per_row * d
        # Vertical extent: (n_rows - 1) * d*sqrt(3)/2 + 2*r = (n_rows-1)*d*sqrt(3)/2 + d
        # So: w = max_per_row * d, h = (n_rows-1)*d*sqrt(3)/2 + d
        # From w: d = w / max_per_row
        # Check h: h_calc = (n_rows-1)*(w/max_per_row)*sqrt(3)/2 + w/max_per_row
        
        d = w / max_per_row
        h_calc = (n_rows - 1) * d * np.sqrt(3) / 2 + d
        
        # Scale to fit both
        if h_calc > h:
            d = h / ((n_rows - 1) * np.sqrt(3) / 2 + 1)
            w_calc = max_per_row * d
        else:
            w_calc = w
        
        r = d / 2
        
        # Place circles
        for row_idx, count in enumerate(rows_pattern):
            y = r + row_idx * d * np.sqrt(3) / 2
            # Offset odd rows
            offset = 0
            if row_idx % 2 == 1:
                offset = d / 2
            
            # Center the row horizontally
            row_width = (count - 1) * d
            x_start = (w_calc - row_width) / 2 + r
            
            for col_idx in range(count):
                x = x_start + col_idx * d
                positions.append([x, y])
                radii.append(r)
        
        return np.array(positions), np.array(radii), w_calc, h
    
    # Try various row patterns that sum to 21
    row_patterns = [
        [4, 5, 4, 4, 4],
        [5, 4, 5, 4, 3],
        [4, 4, 5, 4, 4],
        [3, 5, 5, 5, 3],
        [4, 5, 5, 4, 3],
        [5, 5, 4, 4, 3],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 5, 5, 4],
        [4, 5, 4, 5, 3],
    ]
    
    # Try various aspect ratios
    aspect_ratios = np.linspace(0.5, 1.5, 21)
    
    for pattern in row_patterns:
        if sum(pattern) != N:
            continue
        
        for ar in aspect_ratios:
            positions, radii, w, h = make_hex_config(pattern, ar)
            
            # Validate and compute score
            score = evaluate_config(positions, radii, w, h)
            if score > best_score:
                best_score = score
                best_result = (positions.copy(), radii.copy(), w, h)
            
            # Check time
            if time.time() - start_time > deadline:
                break
        
        if time.time() - start_time > deadline:
            break
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    positions, radii, w, h = best_result
    
    # Local optimization to improve
    def optimize_config(pos, rad, w, h, max_iters=500):
        """Optimize positions to maximize minimum clearance"""
        pos = pos.copy()
        rad = rad.copy()
        n = len(pos)
        
        # Objective: maximize sum of radii, or equivalently, for fixed geometry,
        # try to increase all radii uniformly
        # Simpler: optimize positions to maximize the minimum pairwise distance minus wall distance
        
        def objective(x):
            pos = x.reshape(n, 2)
            min_clearance = float('inf')
            # Wall clearance
            for i in range(n):
                min_clearance = min(min_clearance, pos[i][0], pos[i][1], w - pos[i][0], h - pos[i][1])
            # Pairwise distances
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((pos[i] - pos[j])**2))
                    min_clearance = min(min_clearance, d / 2)
            return -min_clearance
        
        # Gradient-free optimization
        result = minimize(objective, pos.flatten(), method='Nelder-Mead', 
                         options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < objective(pos.flatten()):
            return result.x.reshape(n, 2)
        return pos
    
    # Try to optimize the best configuration
    if time.time() - start_time < deadline - 30:
        try:
            opt_positions = optimize_config(positions, radii, w, h, max_iters=200)
            # Recompute radii from optimized positions
            # The radius is limited by the minimum of wall distances and half pairwise distances
            new_radii = np.zeros(N)
            for i in range(N):
                # Wall clearance
                r_wall = min(opt_positions[i][0], opt_positions[i][1], w - opt_positions[i][0], h - opt_positions[i][1])
                # Pairwise
                r_pair = float('inf')
                for j in range(N):
                    if i != j:
                        d = np.sqrt(np.sum((opt_positions[i] - opt_positions[j])**2))
                        r_pair = min(r_pair, d / 2)
                new_radii[i] = min(r_wall, r_pair)
            
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_score = new_score
                positions = opt_positions
                radii = new_radii
        except:
            pass
    
    # Final validation
    final_score = evaluate_config(positions, radii, w, h)
    if final_score <= 0:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Return in the required format: (x, y, r)
    result = np.column_stack([positions, radii])
    return result

# EVOLVE-BLOCK-END
