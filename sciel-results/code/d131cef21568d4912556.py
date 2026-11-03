"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _grow_and_resolve(circles, W, H, n_grow_steps=200, n_resolve_iters=50, growth_rate=1.01):
    """Iteratively grow radii and resolve collisions by pushing circles apart."""
    circles = circles.copy()
    r = circles[:, 2]
    
    for step in range(n_grow_steps):
        # Grow all radii slightly
        r = r * growth_rate
        
        # Cap radii by boundary constraints
        max_r = np.minimum(np.minimum(circles[:, 0], W - circles[:, 0]),
                          np.minimum(circles[:, 1], H - circles[:, 1]))
        r = np.minimum(r, max_r * 0.999)
        r = np.maximum(r, 1e-6)
        circles[:, 2] = r
        
        # Resolve collisions
        for it in range(n_resolve_iters):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist - 1e-10:
                        if dist < 1e-12:
                            # Circles at same point, push apart randomly
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = (min_dist - dist) / 2 + 1e-6
                            circles[i][0] += push * np.cos(angle)
                            circles[i][1] += push * np.sin(angle)
                            circles[j][0] -= push * np.cos(angle)
                            circles[j][1] -= push * np.sin(angle)
                        else:
                            # Push apart along the line connecting centers
                            overlap = (min_dist - dist) / 2 + 1e-6
                            nx = dx / dist
                            ny = dy / dist
                            circles[i][0] += overlap * nx
                            circles[i][1] += overlap * ny
                            circles[j][0] -= overlap * nx
                            circles[j][1] -= overlap * ny
                        moved = True
            
            # Also push circles that are outside the boundary back inside
            for i in range(N):
                x, y, ri = circles[i]
                if x - ri < 0:
                    circles[i][0] = ri
                if y - ri < 0:
                    circles[i][1] = ri
                if x + ri > W:
                    circles[i][0] = W - ri
                if y + ri > H:
                    circles[i][1] = H - ri
            
            if not moved:
                break
    
    return circles


def _adapt_aspect_ratio(circles, W, H, n_iters=50):
    """Try to adapt the aspect ratio to better fit the circles."""
    # Compute the bounding box of the circles
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    # The circles want a rectangle of size (max_x - min_x, max_y - min_y)
    desired_W = max_x - min_x
    desired_H = max_y - min_y
    
    # We need 2*(W + H) = 4, so W + H = 2
    # Find W, H that best match the desired aspect ratio
    # W/H = desired_W/desired_H, W + H = 2
    # W = 2 * desired_W / (desired_W + desired_H)
    # H = 2 * desired_H / (desired_W + desired_H)
    
    if desired_W + desired_H < 1e-6:
        return W, H
    
    new_W = 2.0 * desired_W / (desired_W + desired_H)
    new_H = 2.0 * desired_H / (desired_W + desired_H)
    
    # Ensure positive dimensions
    new_W = max(new_W, 0.1)
    new_H = max(new_H, 0.1)
    
    # Renormalize to maintain perimeter = 4
    scale = 4.0 / (2.0 * (new_W + new_H))
    new_W *= scale
    new_H *= scale
    
    return new_W, new_H


def _initial_grid(W, H, n_circles=N):
    """Create a grid-based initial configuration with small circles."""
    side = int(np.ceil(np.sqrt(n_circles)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n_circles:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) * 0.3  # Start small
            cells.append([cx, cy, r])
    return np.array(cells[:n_circles])


def _optimize_with_growth(W, H, rng, n_grow_steps=300, n_resolve_iters=80, growth_rate=1.015):
    """Main optimization: start with grid, grow radii with collision resolution, adapt aspect ratio."""
    circles = _initial_grid(W, H)
    
    current_W, current_H = W, H
    
    for outer in range(20):
        # Grow and resolve
        circles = _grow_and_resolve(circles, current_W, current_H, 
                                     n_grow_steps=n_grow_steps, 
                                     n_resolve_iters=n_resolve_iters,
                                     growth_rate=growth_rate)
        
        # Clip to valid range
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], current_W - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], current_H - circles[:, 2])
        circles[:, 2] = np.clip(circles[:, 2], 1e-6, min(current_W, current_H) / 2)
        
        # Adapt aspect ratio
        new_W, new_H = _adapt_aspect_ratio(circles, current_W, current_H)
        
        # Scale circles to fit new rectangle
        scale_x = new_W / current_W
        scale_y = new_H / current_H
        circles[:, 0] *= scale_x
        circles[:, 1] *= scale_y
        circles[:, 2] *= np.minimum(scale_x, scale_y)
        
        current_W, current_H = new_W, new_H
    
    return circles, current_W, current_H


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W = 1.0
    best_H = 1.0
    
    rng = np.random.default_rng(42)
    
    # Try different initial aspect ratios
    aspect_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        # Run growth-based optimization
        circles, final_W, final_H = _optimize_with_growth(W, H, rng)
        
        # Verify validity
        if _check_validity(circles, final_W, final_H):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W = final_W
                best_H = final_H
    
    # Try a few more random restarts if time permits
    if time.time() - start_time < DEADLINE - 10:
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random aspect ratio
            ar = rng.uniform(0.5, 2.5)
            W = 2.0 * ar / (ar + 1.0)
            H = 2.0 / (ar + 1.0)
            
            # Random initial positions
            circles = np.zeros((N, 3))
            for i in range(N):
                r = rng.uniform(0.02, 0.1)
                x = rng.uniform(r, W - r)
                y = rng.uniform(r, H - r)
                circles[i] = [x, y, r]
            
            # Grow and resolve
            circles, final_W, final_H = _optimize_with_growth(W, H, rng, 
                                                              n_grow_steps=200,
                                                              n_resolve_iters=60,
                                                              growth_rate=1.02)
            
            if _check_validity(circles, final_W, final_H):
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circles.copy()
                    best_W = final_W
                    best_H = final_H
    
    # Final fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    # Final validity check; if invalid, use safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
