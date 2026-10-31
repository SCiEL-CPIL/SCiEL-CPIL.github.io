"""
Circle packing in a perimeter-4 rectangle.
Uses force-directed relaxation with adaptive radii.
Circles repel each other and are attracted to maximize radius while staying in bounds.
"""

import numpy as np
import time
import random

N = 21
DEADLINE = 110.0  # Safety margin before 120s limit


def _compute_max_radii(centers, w, h):
    """Compute the maximum possible radius for each circle given centers and box."""
    n = len(centers)
    max_radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        # Distance to walls
        r = min(x, w - x, y, h - y)
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                r = min(r, d / 2.0)
        max_radii[i] = max(0.0, r)
    return max_radii


def _force_directed_relax(centers, radii, w, h, max_steps=2000, learning_rate=0.005):
    """
    Force-directed relaxation:
    - Repulsive forces between overlapping circles
    - Attractive force to increase radii (via gradient of sum of radii)
    - Boundary forces to keep circles inside
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    # Track best configuration
    best_score = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    for step in range(max_steps):
        # Compute forces
        forces = np.zeros_like(centers)
        radius_forces = np.zeros(n)
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Add small random perturbation
                    d = np.random.randn(2) * 0.01
                    dist = np.linalg.norm(d)
                
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    force_mag = overlap * 10.0
                    force_dir = d / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
                    
                    # Also push radii down slightly when overlapping
                    radius_forces[i] -= overlap * 2.0
                    radius_forces[j] -= overlap * 2.0
        
        # Boundary forces: push circles away from walls
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i][0] += (r - x) * 20.0
            # Right wall
            if x > w - r:
                forces[i][0] -= (x - (w - r)) * 20.0
            # Bottom wall
            if y < r:
                forces[i][1] += (r - y) * 20.0
            # Top wall
            if y > h - r:
                forces[i][1] -= (y - (h - r)) * 20.0
        
        # Update centers
        centers += learning_rate * forces
        # Clamp to box (with small margin)
        for i in range(n):
            centers[i][0] = np.clip(centers[i][0], 0, w)
            centers[i][1] = np.clip(centers[i][1], 0, h)
        
        # Update radii: try to grow, but limited by geometry
        for i in range(n):
            # Compute max possible radius at current position
            x, y = centers[i]
            max_r = min(x, w - x, y, h - y)
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d / 2.0)
            
            # Grow toward max_r
            target_r = max(0.0, max_r)
            radii[i] += learning_rate * (target_r - radii[i]) * 5.0
            radii[i] = max(0.0, min(radii[i], target_r))
        
        # Track best
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            best_radii = radii.copy()
        
        # Early stopping if no improvement for many steps
        if step % 200 == 199 and current_score <= best_score * 1.0001:
            break
    
    # Final: compute exact max radii for the best centers
    final_radii = _compute_max_radii(best_centers, w, h)
    final_score = np.sum(final_radii)
    
    return final_score, best_centers, final_radii


def _hexagonal_layout(w, h, n_circles):
    """Create a hexagonal packing layout as initial guess."""
    centers = []
    # Estimate radius from area
    area = w * h
    avg_area_per_circle = area / n_circles
    est_r = np.sqrt(avg_area_per_circle / np.pi) * 0.8
    
    # Hexagonal spacing
    dx = est_r * 2.0
    dy = est_r * np.sqrt(3)
    
    row = 0
    while len(centers) < n_circles:
        offset = (dx / 2.0) if (row % 2 == 1) else 0.0
        col = 0
        while len(centers) < n_circles:
            x = offset + col * dx
            y = row * dy
            if x < w and y < h:
                centers.append([x, y])
            else:
                break
            col += 1
        row += 1
    
    centers = np.array(centers[:n_circles])
    radii = np.full(n_circles, est_r * 0.5)
    return centers, radii


def _grid_layout(w, h, n_circles):
    """Create a grid layout as initial guess."""
    # Find best grid dimensions
    best_cols = 1
    best_rows = n_circles
    best_aspect_diff = float('inf')
    
    for cols in range(1, n_circles + 1):
        rows = (n_circles + cols - 1) // cols
        if cols * rows >= n_circles:
            # Aspect ratio of grid
            grid_aspect = cols / rows
            box_aspect = w / h
            diff = abs(grid_aspect - box_aspect)
            if diff < best_aspect_diff:
                best_aspect_diff = diff
                best_cols = cols
                best_rows = rows
    
    centers = []
    for i in range(best_rows):
        for j in range(best_cols):
            if len(centers) < n_circles:
                x = (j + 0.5) * w / best_cols
                y = (i + 0.5) * h / best_rows
                centers.append([x, y])
    
    centers = np.array(centers[:n_circles])
    cell_w = w / best_cols
    cell_h = h / best_rows
    est_r = min(cell_w, cell_h) / 2.0 * 0.8
    radii = np.full(n_circles, est_r)
    return centers, radii


def _random_layout(w, h, n_circles):
    """Create a random layout as initial guess."""
    centers = np.random.rand(n_circles, 2) * np.array([w, h])
    # Small initial radii
    radii = np.full(n_circles, 0.03)
    return centers, radii


def construct():
    start_time = time.time()
    
    best_score = 0.0
    best_result = None
    
    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 1.33, 0.75, 1.25]
    
    for idx, aspect in enumerate(aspect_ratios):
        if time.time() > DEADLINE - 5:
            break
        
        w = aspect
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            continue
        
        # Try different initializations
        inits = []
        
        # Grid
        c_grid, r_grid = _grid_layout(w, h, N)
        inits.append((c_grid, r_grid))
        
        # Hexagonal
        c_hex, r_hex = _hexagonal_layout(w, h, N)
        inits.append((c_hex, r_hex))
        
        # Random (2 variants)
        c_rand1, r_rand1 = _random_layout(w, h, N)
        inits.append((c_rand1, r_rand1))
        
        c_rand2, r_rand2 = _random_layout(w, h, N)
        inits.append((c_rand2, r_rand2))
        
        for init_idx, (init_centers, init_radii) in enumerate(inits):
            if time.time() > DEADLINE - 3:
                break
            
            # Run force-directed relaxation
            score, centers, radii = _force_directed_relax(
                init_centers, init_radii, w, h, 
                max_steps=1500, learning_rate=0.004
            )
            
            if score > best_score:
                best_score = score
                best_result = (centers, radii, w)
            
            # Check time
            if time.time() > DEADLINE - 5:
                break
    
    # If we found a good result, return it
    if best_result is not None and best_score > 0:
        centers, radii, w = best_result
        result = np.column_stack((centers, radii))
        return result
    
    # Fallback: simple grid in a square
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    return np.array(cells[:N])
