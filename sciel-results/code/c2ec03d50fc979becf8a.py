import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try different grid configurations and aspect ratios
    # For m x n grid, aspect ratio w/h ≈ m/n
    # Perimeter 4: 2(w+h) = 4, so w + h = 2
    
    # Various grid layouts for 21 circles
    # We'll use grids with at least 21 cells and select 21
    configs = []
    for m in range(3, 8):
        for n in range(3, 8):
            if m * n >= N:
                configs.append((m, n))
    
    # Remove duplicates and keep reasonable ones
    configs = list(set(configs))
    configs.sort(key=lambda x: abs(x[0]/x[1] - 1.0))  # prefer near-square first
    
    # Also try some non-grid approaches with different aspect ratios
    aspect_ratios = []
    for m, n in configs:
        ar = m / n
        aspect_ratios.append(ar)
    # Add some extra aspect ratios
    aspect_ratios.extend([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0])
    aspect_ratios = sorted(set([a for a in aspect_ratios if 0.3 < a < 3.0]))
    
    def solve_for_aspect(w, h, m, n, seed_offset=0):
        """Solve for circles in rectangle w x h using m x n grid init"""
        if w <= 0 or h <= 0:
            return None, 0.0
        
        # Initialize circles in grid
        positions = []
        for i in range(m):
            for j in range(n):
                x = (i + 0.5) * w / m
                y = (j + 0.5) * h / n
                positions.append([x, y])
        
        positions = np.array(positions[:N], dtype=float)
        
        # Initial radii: limited by grid spacing and boundary
        min_spacing_x = w / m
        min_spacing_y = h / n
        init_r = min(min_spacing_x, min_spacing_y) * 0.45
        radii = np.full(N, init_r)
        
        # Variable vector: [x1, y1, r1, x2, y2, r2, ...]
        def unpack(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            return xs, ys, rs
        
        def objective(v):
            xs, ys, rs = unpack(v)
            
            # Penalty for boundary violations
            pen = 0.0
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    pen += (xs[i] - rs[i]) ** 2
                if xs[i] + rs[i] > w:
                    pen += (xs[i] + rs[i] - w) ** 2
                if ys[i] - rs[i] < 0:
                    pen += (ys[i] - rs[i]) ** 2
                if ys[i] + rs[i] > h:
                    pen += (ys[i] + rs[i] - h) ** 2
            
            # Penalty for overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        pen += (min_dist - dist) ** 2
            
            # Negative sum of radii (we minimize)
            val = -np.sum(rs) + 1000.0 * pen
            return val
        
        # Initial vector
        v0 = np.zeros(N * 3)
        for i in range(N):
            v0[i * 3] = positions[i, 0]
            v0[i * 3 + 1] = positions[i, 1]
            v0[i * 3 + 2] = radii[i]
        
        # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        r_max = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        # Run Nelder-Mead
        try:
            result = minimize(
                objective, v0, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            xs, ys, rs = unpack(result.x)
            
            # Validate
            valid = True
            for i in range(N):
                if xs[i] - rs[i] < -1e-6 or xs[i] + rs[i] > w + 1e-6:
                    valid = False
                    break
                if ys[i] - rs[i] < -1e-6 or ys[i] + rs[i] > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                value = np.sum(rs)
                # Clip to be safe
                for i in range(N):
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                
                # Final validation after clipping
                valid2 = True
                for i in range(N):
                    if xs[i] - rs[i] < -1e-9 or xs[i] + rs[i] > w + 1e-9:
                        valid2 = False
                        break
                    if ys[i] - rs[i] < -1e-9 or ys[i] + rs[i] > h + 1e-9:
                        valid2 = False
                        break
                if valid2:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < rs[i] + rs[j] - 1e-9:
                                valid2 = False
                                break
                        if not valid2:
                            break
                
                if valid2:
                    return np.column_stack([xs, ys, rs]), value
            else:
                # Try to repair: reduce radii slightly
                for i in range(N):
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                # Check again
                valid3 = True
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < rs[i] + rs[j] - 1e-9:
                            valid3 = False
                            break
                    if not valid3:
                        break
                if valid3:
                    value = np.sum(rs)
                    return np.column_stack([xs, ys, rs]), value
        
        except Exception:
            pass
        
        return None, 0.0
    
    # Try each configuration
    for m, n in configs:
        if time.time() > deadline:
            break
        
        # Optimal aspect ratio for m x n grid: w/h = m/n
        # But we can also try nearby aspect ratios
        base_ar = m / n
        
        # Try a few aspect ratios around the base
        for ar_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time.time() > deadline:
                break
            
            ar = base_ar * ar_factor
            w = 2.0 * ar / (1.0 + ar)
            h = 2.0 / (1.0 + ar)
            
            result, value = solve_for_aspect(w, h, m, n)
            if result is not None and value > best_value:
                best_value = value
                best_result = result
    
    # Also try some direct aspect ratio sweeps with a generic grid
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Choose grid size based on aspect ratio
        if ar >= 1:
            m = int(np.round(np.sqrt(N * ar)))
            n = int(np.round(np.sqrt(N / ar)))
        else:
            m = int(np.round(np.sqrt(N / ar)))
            n = int(np.round(np.sqrt(N * ar)))
        
        # Ensure m*n >= N
        while m * n < N:
            if m <= n:
                m += 1
            else:
                n += 1
        
        result, value = solve_for_aspect(w, h, m, n)
        if result is not None and value > best_value:
            best_value = value
            best_result = result
    
    # Fallback: use seed solution
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
