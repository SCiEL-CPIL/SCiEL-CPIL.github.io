import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_r = 0.0
    best_x = None
    best_y = None
    best_W = 1.0
    best_H = 1.0

    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N]
        W = v[2*N+1]
        H = 2.0 - W
        return x, y, r, W, H

    def penalty_objective(v):
        x, y, r, W, H = unpack(v)
        
        # Penalty for circle-circle overlaps
        # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # overlap = max(0, 2r - d_ij)
        # We use squared overlap to make it smooth and differentiable
        pen_cc = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                # Avoid sqrt if possible? No, we need distance for gradient if using analytical, 
                # but L-BFGS-B uses numerical gradients or we can provide them.
                # Numerical gradients are fine for this size.
                d = np.sqrt(d_sq) if d_sq > 0 else 1e-10
                min_d = 2.0 * r
                if d < min_d:
                    pen_cc += (min_d - d)**2
        
        # Penalty for wall violations
        pen_w = 0.0
        for i in range(N):
            # x bounds: [r, W-r]
            if x[i] < r:
                pen_w += (r - x[i])**2
            elif x[i] > W - r:
                pen_w += (x[i] - (W - r))**2
            
            # y bounds: [r, H-r]
            if y[i] < r:
                pen_w += (r - y[i])**2
            elif y[i] > H - r:
                pen_w += (y[i] - (H - r))**2
        
        total_penalty = pen_cc + pen_w
        # We want to maximize r, so minimize -r.
        # Add penalty with a large weight.
        weight = 100.0
        return -r + weight * total_penalty

    def make_initial_grid(W0, H0):
        # Simple grid initialization
        side = int(np.ceil(np.sqrt(N)))
        x_list = []
        y_list = []
        for i in range(side):
            for j in range(side):
                if len(x_list) < N:
                    x_list.append((i + 0.5) * (W0 / side))
                    y_list.append((j + 0.5) * (H0 / side))
        x0 = np.array(x_list[:N])
        y0 = np.array(y_list[:N])
        # Estimate initial radius
        r0 = 0.4 * min(W0, H0) / side
        v0 = np.concatenate([x0, y0, [r0, W0]])
        return v0

    def make_initial_hex(W0, H0):
        # Hexagonal-like initialization
        # This is often better for packing
        x_list = []
        y_list = []
        # Roughly 5 rows
        rows = [5, 4, 5, 4, 3]
        y_spacing = H0 / 6.0
        x_spacing = W0 / 5.5
        idx = 0
        for row_idx, count in enumerate(rows):
            y = y_spacing * (row_idx + 1)
            offset = 0 if row_idx % 2 == 0 else x_spacing / 2
            for col in range(count):
                if idx >= N:
                    break
                x = x_spacing * (col + 1) + offset
                x_list.append(x)
                y_list.append(y)
                idx += 1
        # Pad if needed
        while len(x_list) < N:
            x_list.append(W0 / 2)
            y_list.append(H0 / 2)
        
        x0 = np.array(x_list[:N])
        y0 = np.array(y_list[:N])
        r0 = 0.15
        v0 = np.concatenate([x0, y0, [r0, W0]])
        return v0

    def make_random_initial(seed, W0):
        H0 = 2.0 - W0
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.1 * W0, 0.9 * W0, N)
        y0 = rng.uniform(0.1 * H0, 0.9 * H0, N)
        r0 = 0.1
        v0 = np.concatenate([x0, y0, [r0, W0]])
        return v0

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for W0 in aspect_ratios:
        if time.time() > deadline:
            break
        H0 = 2.0 - W0
        
        # Generate initializations for this aspect ratio
        inits = [
            make_initial_grid(W0, H0),
            make_initial_hex(W0, H0),
            make_random_initial(1, W0),
            make_random_initial(2, W0),
            make_random_initial(3, W0)
        ]

        for v0 in inits:
            if time.time() > deadline:
                break
            
            try:
                # Bounds
                # x: [0, 2], y: [0, 2], r: [0.01, 1.0], W: [0.5, 1.5]
                bounds = [(0, 2.0)] * N + [(0, 2.0)] * N + [(0.01, 1.0)] + [(0.5, 1.5)]
                
                res = minimize(
                    penalty_objective, 
                    v0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                x, y, r, W, H = unpack(res.x)
                
                # Validate the solution
                valid = True
                if r <= 0 or W <= 0 or H <= 0:
                    valid = False
                else:
                    for k in range(N):
                        if x[k] < r - 1e-9 or x[k] > W - r + 1e-9 or y[k] < r - 1e-9 or y[k] > H - r + 1e-9:
                            valid = False
                            break
                    if valid:
                        for k in range(N):
                            for l in range(k+1, N):
                                if (x[k]-x[l])**2 + (y[k]-y[l])**2 < (2*r)**2 - 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                
                if valid and r > best_r:
                    best_r = r
                    best_x = x
                    best_y = y
                    best_W = W
                    best_H = H
                    
            except Exception:
                continue

    if best_x is not None:
        result = np.column_stack((best_x, best_y, np.full(N, best_r)))
        return result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
