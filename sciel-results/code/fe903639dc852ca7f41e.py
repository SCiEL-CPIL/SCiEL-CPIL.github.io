"""Circle packing in perimeter-4 rectangle using force-directed relaxation."""
import numpy as np
import time

N = 21

def _force_relax(cx, cy, r, W, H, max_iters=5000, dt=0.001, deadline=None):
    """
    Force-directed relaxation: circles repel each other and are attracted to fill space.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Reset forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Pairwise repulsion: push circles apart if too close
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force = overlap * 10.0  # stiffness
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    # Also try to grow radii
                    fr[i] += overlap * 0.5
                    fr[j] += overlap * 0.5
                else:
                    # Weak attraction to encourage packing
                    force = 0.1 * (dist - min_dist)
                    fx[i] -= force * dx / dist
                    fy[i] -= force * dy / dist
                    fx[j] += force * dx / dist
                    fy[j] += force * dy / dist
        
        # Wall constraints: push circles away from walls
        for i in range(n):
            # Left wall
            if cx[i] - r[i] < 0:
                fx[i] += (r[i] - cx[i]) * 20.0
                fr[i] -= (r[i] - cx[i]) * 5.0
            # Right wall
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20.0
                fr[i] -= (cx[i] + r[i] - W) * 5.0
            # Bottom wall
            if cy[i] - r[i] < 0:
                fy[i] += (r[i] - cy[i]) * 20.0
                fr[i] -= (r[i] - cy[i]) * 5.0
            # Top wall
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20.0
                fr[i] -= (cy[i] + r[i] - H) * 5.0
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Update radii (grow if space allows)
        r += dt * fr * 0.1
        
        # Enforce minimum radius
        r = np.maximum(r, 0.01)
        
        # Enforce wall constraints (clamp positions)
        for i in range(n):
            cx[i] = np.clip(cx[i], r[i], W - r[i])
            cy[i] = np.clip(cy[i], r[i], H - r[i])
    
    return cx, cy, r

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if the packing is valid."""
    n = len(cx)
    # Wall constraints
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True

def _hex_init(W, H, n, seed=0):
    """Generate hexagonal grid initialization scaled to fit W x H."""
    rng = np.random.default_rng(seed)
    
    # Estimate radius to fit N circles in W x H
    r_est = np.sqrt(0.9 * W * H / (n * np.pi))
    
    # Create a grid of points
    cols = int(np.ceil(np.sqrt(n * W / H)))
    if cols < 1:
        cols = 1
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    for row in range(rows):
        for col in range(cols):
            if len(xs) >= n:
                break
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            xs.append(x)
            ys.append(y)
            
    while len(xs) < n:
        xs.append(rng.uniform(0, max(xs) if xs else 1))
        ys.append(rng.uniform(0, max(ys) if ys else 1))
        
    xs = np.array(xs[:n])
    ys = np.array(ys[:n])
    
    # Scale to fit W and H
    cx = (xs - np.mean(xs)) * (W / (np.max(xs) - np.min(xs) + 1e-6)) + W/2
    cy = (ys - np.mean(ys)) * (H / (np.max(ys) - np.min(ys) + 1e-6)) + H/2
    
    cx = np.clip(cx, 0.01, W - 0.01)
    cy = np.clip(cy, 0.01, H - 0.01)
    
    # Initial radii: small to ensure no overlap
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            min_dist = min(min_dist, d)
            
    r = np.full(n, min_dist * 0.4)
    
    cx += rng.uniform(-0.02, 0.02, n)
    cy += rng.uniform(-0.02, 0.02, n)
    cx = np.clip(cx, 0.01, W - 0.01)
    cy = np.clip(cy, 0.01, H - 0.01)
    
    return cx, cy, r

def _optimize_aspect(W, H, deadline, seed=0):
    """Optimize for a specific aspect ratio using force-directed relaxation."""
    if time.time() > deadline:
        return None, -float('inf')
        
    cx, cy, r = _hex_init(W, H, N, seed=seed)
    
    # Run force-directed relaxation
    cx_opt, cy_opt, r_opt = _force_relax(cx, cy, r, W, H, max_iters=3000, dt=0.002, deadline=deadline)
    
    # Validate
    if _is_valid(cx_opt, cy_opt, r_opt, W, H):
        return (cx_opt, cy_opt, r_opt), np.sum(r_opt)
    
    return None, -float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_val = -float('inf')
    best_config = None
    
    # Try different aspect ratios
    # Perimeter 4 => W + H = 2
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    np.random.seed(42)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
            
        # Perform a few restarts
        num_restarts = 2 if time.time() < deadline - 20 else 1
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            config, val = _optimize_aspect(W, H, deadline, seed=restart)
            
            if val > best_val:
                best_val = val
                best_config = config
                
    if best_config is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r = best_config
    return np.column_stack([cx, cy, r])
