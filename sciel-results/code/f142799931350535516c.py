"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time

# EVOLVE-BLOCK-START

def _force_directed_relax(xs, ys, rs, w, h, max_iter=200, lr=0.01):
    """
    Force-directed relaxation: circles repel each other and walls,
    while we try to grow radii.
    """
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    rs = np.array(rs, dtype=float)
    
    for iteration in range(max_iter):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Circle-circle repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx = 0.01 * (np.random.random() - 0.5)
                    dy = 0.01 * (np.random.random() - 0.5)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    # Reduce radii when overlapping
                    fr[i] -= 0.5
                    fr[j] -= 0.5
        
        # Wall forces
        for i in range(N):
            # Left wall
            if xs[i] < rs[i]:
                overlap = rs[i] - xs[i]
                fx[i] += overlap * 10.0
                fr[i] -= 0.5
            # Right wall
            if xs[i] > w - rs[i]:
                overlap = xs[i] - (w - rs[i])
                fx[i] -= overlap * 10.0
                fr[i] -= 0.5
            # Bottom wall
            if ys[i] < rs[i]:
                overlap = rs[i] - ys[i]
                fy[i] += overlap * 10.0
                fr[i] -= 0.5
            # Top wall
            if ys[i] > h - rs[i]:
                overlap = ys[i] - (h - rs[i])
                fy[i] -= overlap * 10.0
                fr[i] -= 0.5
        
        # Growth force: try to increase radii
        growth_rate = 0.005
        fr += growth_rate
        
        # Update positions and radii
        xs += lr * fx
        ys += lr * fy
        rs += lr * fr
        
        # Clamp to valid range
        for i in range(N):
            rs[i] = max(rs[i], 0.01)
            xs[i] = np.clip(xs[i], 0, w)
            ys[i] = np.clip(ys[i], 0, h)
    
    return xs, ys, rs

def _evaluate(xs, ys, rs, w, h):
    """Check validity and compute sum of radii."""
    # Check wall constraints
    for i in range(N):
        if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
            return -1.0
        if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
            return -1.0
    
    # Check circle-circle constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return -1.0
    
    return np.sum(rs)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_sum_r = 0.0
    best_result = None
    
    # Sample aspect ratios: w in [0.5, 1.0] (h = 2-w, so h in [1.0, 1.5])
    w_values = np.linspace(0.5, 1.0, 15)
    
    for w in w_values:
        if time.time() > deadline:
            break
        
        h = 2.0 - w
        
        # Multiple initialization strategies
        inits = []
        
        # 1. Grid initialization
        for rows in [3, 4, 5, 6]:
            for cols in [3, 4, 5, 6, 7]:
                if rows * cols >= N:
                    xs_grid = []
                    ys_grid = []
                    for i in range(rows):
                        for j in range(cols):
                            x = (j + 0.5) * w / cols
                            y = (i + 0.5) * h / rows
                            xs_grid.append(x)
                            ys_grid.append(y)
                    if len(xs_grid) >= N:
                        xs_grid = np.array(xs_grid[:N])
                        ys_grid = np.array(ys_grid[:N])
                        
                        # Estimate initial radius
                        min_dist = np.inf
                        for i in range(N):
                            for j in range(i+1, N):
                                d = np.sqrt((xs_grid[i]-xs_grid[j])**2 + (ys_grid[i]-ys_grid[j])**2)
                                min_dist = min(min_dist, d)
                        for i in range(N):
                            d_wall = min(xs_grid[i], w-xs_grid[i], ys_grid[i], h-ys_grid[i])
                            min_dist = min(min_dist, d_wall)
                        
                        r_init = min_dist / 2.0 * 0.8
                        rs_grid = np.full(N, r_init)
                        inits.append((xs_grid.copy(), ys_grid.copy(), rs_grid.copy()))
        
        # 2. Random initialization
        np.random.seed(42)
        for _ in range(5):
            xs_rand = np.random.uniform(0.1, w-0.1, N)
            ys_rand = np.random.uniform(0.1, h-0.1, N)
            rs_rand = np.full(N, 0.1)
            inits.append((xs_rand, ys_rand, rs_rand))
        
        # 3. Hexagonal-ish initialization
        hex_inits = []
        for offset in [0, 0.25, 0.5]:
            xs_hex = []
            ys_hex = []
            row = 0
            while len(xs_hex) < N:
                for col in range(10):
                    x = (col + 0.5 + (0.5 if row % 2 else 0)) * w / 10
                    y = (row + 0.5) * h / 8
                    if 0 <= x <= w and 0 <= y <= h:
                        xs_hex.append(x)
                        ys_hex.append(y)
                row += 1
            if len(xs_hex) >= N:
                xs_hex = np.array(xs_hex[:N])
                ys_hex = np.array(ys_hex[:N])
                min_dist = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs_hex[i]-xs_hex[j])**2 + (ys_hex[i]-ys_hex[j])**2)
                        min_dist = min(min_dist, d)
                for i in range(N):
                    d_wall = min(xs_hex[i], w-xs_hex[i], ys_hex[i], h-ys_hex[i])
                    min_dist = min(min_dist, d_wall)
                r_init = min_dist / 2.0 * 0.8
                rs_hex = np.full(N, r_init)
                hex_inits.append((xs_hex.copy(), ys_hex.copy(), rs_hex.copy()))
        inits.extend(hex_inits)
        
        # Run force-directed relaxation for each initialization
        for xs_init, ys_init, rs_init in inits:
            if time.time() > deadline:
                break
            
            xs, ys, rs = _force_directed_relax(
                xs_init, ys_init, rs_init, w, h,
                max_iter=300, lr=0.005
            )
            
            val = _evaluate(xs, ys, rs, w, h)
            if val > best_sum_r:
                best_sum_r = val
                best_result = np.column_stack([xs, ys, rs])
    
    if best_result is None:
        # Fallback: simple grid in a square-ish rectangle
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) * w / side
                y = (i + 0.5) * h / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
