"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_value(circles):
    """Compute sum of radii and check validity. Returns (value, valid)."""
    if circles is None or len(circles) != N:
        return 0.0, False
    if not np.all(np.isfinite(circles)):
        return 0.0, False
    radii = circles[:, 2]
    if np.any(radii <= 0):
        return 0.0, False
    if np.any(radii > 1.0):
        return 0.0, False
    
    # Bounding box
    min_x = np.min(circles[:, 0] - radii)
    max_x = np.max(circles[:, 0] + radii)
    min_y = np.min(circles[:, 1] - radii)
    max_y = np.max(circles[:, 1] + radii)
    
    W = max_x - min_x
    H = max_y - min_y
    
    if W <= 0 or H <= 0:
        return 0.0, False
    if 2 * (W + H) > 4.0 + 1e-9:
        return 0.0, False
    
    # Check non-overlap: distance >= r_i + r_j for all pairs
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = radii
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    
    return np.sum(radii), True


def _make_hex_layout(n, W, H, rows, cols):
    """Create a hexagonal layout of n circles in W x H rectangle."""
    circles = []
    for row in range(rows):
        for col in range(cols):
            if len(circles) >= n:
                break
            # Hex offset for odd rows
            x_offset = 0.5 * W / cols if row % 2 == 1 else 0.0
            cx = (col + 0.5) * W / cols + x_offset
            cy = (row + 0.5) * H / rows
            r = min(W / cols, H / rows) * 0.42
            circles.append([cx, cy, r])
    return np.array(circles[:n])


def _resolve_overlaps(circles, W, H, n_iter=200, step_size=0.01):
    """Resolve overlaps by pushing circles apart iteratively."""
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(n_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    # Push apart along the line connecting centers
                    push = overlap * 0.5 * step_size
                    nx, ny = dx / dist, dy / dist
                    circles[i, 0] += nx * push
                    circles[i, 1] += ny * push
                    circles[j, 0] -= nx * push
                    circles[j, 1] -= ny * push
                    moved = True
                elif dist <= 1e-10:
                    # Random small push if exactly overlapping
                    circles[i, 0] += step_size * 0.1
                    circles[j, 0] -= step_size * 0.1
                    moved = True
        
        # Clamp to bounds
        for i in range(n):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
        
        if not moved:
            break
    
    return circles


def _grow_radii(circles, W, H, growth_rate=0.002, max_iter=500):
    """Iteratively grow radii while resolving overlaps."""
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_iter):
        # Check if we can grow
        can_grow = True
        for i in range(n):
            # Check boundary
            if circles[i, 2] + growth_rate > W - circles[i, 0] or \
               circles[i, 2] + growth_rate > circles[i, 0] or \
               circles[i, 2] + growth_rate > H - circles[i, 1] or \
               circles[i, 2] + growth_rate > circles[i, 1]:
                can_grow = False
                break
        
        if not can_grow:
            break
        
        # Try to grow all radii
        old_radii = circles[:, 2].copy()
        circles[:, 2] += growth_rate
        
        # Check for overlaps
        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            # Resolve overlaps
            circles = _resolve_overlaps(circles, W, H, n_iter=50, step_size=0.02)
            
            # Re-check if resolution was successful
            still_overlapping = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                        still_overlapping = True
                        break
                if still_overlapping:
                    break
            
            if still_overlapping:
                # Revert radii growth
                circles[:, 2] = old_radii
                break
        
        # Clamp to bounds
        for i in range(n):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
            circles[i, 2] = np.clip(circles[i, 2], 1e-6, min(W, H)/2)
    
    return circles


def _pack_for_aspect(W, H, rows, cols, time_limit):
    """Pack circles for a given aspect ratio using radius growth approach."""
    start = time.time()
    
    # Create initial hex layout
    circles = _make_hex_layout(N, W, H, rows, cols)
    
    # Resolve initial overlaps
    circles = _resolve_overlaps(circles, W, H, n_iter=100, step_size=0.01)
    
    # Grow radii
    circles = _grow_radii(circles, W, H, growth_rate=0.003, max_iter=300)
    
    # Final validation
    val, valid = _compute_value(circles)
    if valid:
        return val, circles
    return 0.0, None


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_val = 0.0
    best_circles = None
    
    # Try different aspect ratios and grid configurations
    # Perimeter = 2(W+H) = 4 => W+H = 2
    aspect_ratios = [
        (1.0, 1.0),       # square
        (1.2, 0.8),
        (0.8, 1.2),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.1, 0.9),
        (0.9, 1.1),
    ]
    
    # Grid configurations (rows, cols) that give at least N circles
    grid_configs = [
        (4, 6),  # 24 cells
        (6, 4),
        (5, 5),  # 25 cells
        (3, 7),  # 21 exactly
        (7, 3),
        (4, 5),  # 20, close
        (5, 4),
        (6, 4),
        (4, 6),
    ]
    
    # Try configurations
    for W, H in aspect_ratios:
        if time.time() > deadline:
            break
        
        for rows, cols in grid_configs:
            if time.time() > deadline:
                break
            if rows * cols < N:
                continue
            
            time_per_config = max(2.0, (deadline - time.time()) / 5.0)
            val, circles = _pack_for_aspect(W, H, rows, cols, time_per_config)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
    
    # Final fallback: if nothing valid, use simple grid
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _make_hex_layout(N, W, H, 4, 6)
        best_circles = _resolve_overlaps(best_circles, W, H, n_iter=50, step_size=0.01)
        val, valid = _compute_value(best_circles)
        if not valid:
            # Very conservative fallback
            W, H = 1.0, 1.0
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.35 / side) for i in range(side) for j in range(side)]
            best_circles = np.array(cells[:N])
    
    # Ensure final result is valid
    val, valid = _compute_value(best_circles)
    if not valid:
        # Last resort
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.3 / side) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles

# EVOLVE-BLOCK-END
