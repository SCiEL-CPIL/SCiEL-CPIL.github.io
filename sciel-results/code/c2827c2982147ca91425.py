"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s kill

    def evaluate(r, h, centers, radii):
        """Compute objective: sum of radii, given rect dims and circle params."""
        # Check constraints
        # Wall constraints
        for i in range(len(centers)):
            cx, cy = centers[i]
            ri = radii[i]
            if cx - ri < -1e-9 or cx + ri > r + 1e-9:
                return -1
            if cy - ri < -1e-9 or cy + ri > h + 1e-9:
                return -1
        # Pairwise constraints
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return -1
        return np.sum(radii)

    def pack_with_layout(n_cols, n_rows, extra_circles=0):
        """Try a grid-based layout and optimize."""
        # Rectangle with perimeter 4: 2*(w+h)=4 => w+h=2
        # Start with square-ish
        w = 1.0
        h = 1.0
        # Adjust based on grid aspect ratio
        if n_rows > 0 and n_cols > 0:
            h = w * n_rows / n_cols
            # Enforce w + h = 2
            scale = 2.0 / (w + h)
            w *= scale
            h *= scale
        
        # Initial centers on grid
        centers = []
        for i in range(n_cols):
            for j in range(n_rows):
                cx = (i + 0.5) * w / n_cols
                cy = (j + 0.5) * h / n_rows
                centers.append([cx, cy])
        
        # Add extra circles if needed
        while len(centers) < N:
            # Place extra circles in gaps
            idx = len(centers)
            cx = w * (0.5 + 0.3 * np.cos(idx * 2.399))
            cy = h * (0.5 + 0.3 * np.sin(idx * 2.399))
            centers.append([cx, cy])
        
        centers = centers[:N]
        centers = np.array(centers)
        
        # Initial radii: min distance to walls and neighbors / 2
        radii = np.zeros(N)
        for i in range(N):
            # Distance to walls
            d_wall = min(centers[i][0], w - centers[i][0], centers[i][1], h - centers[i][1])
            radii[i] = 0.3 * d_wall
        
        # Optimize: variables are centers (2N) and radii (N), total 3N = 63
        # Objective: maximize sum(radii) => minimize -sum(radii)
        def objective(x):
            c = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            # Wall constraints
            for i in range(N):
                if c[i][0] - r[i] < -1e-9 or c[i][0] + r[i] > w + 1e-9:
                    return 1e6
                if c[i][1] - r[i] < -1e-9 or c[i][1] + r[i] > h + 1e-9:
                    return 1e6
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    dx = c[i][0] - c[j][0]
                    dy = c[i][1] - c[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-9:
                        return 1e6 + (r[i] + r[j] - dist)
            return -np.sum(r)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # cx
            bounds.append((0, h))  # cy
        for i in range(N):
            bounds.append((0, min(w, h)/2))  # radius
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12})
            if res.success or res.fun < 1e5:
                c = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:]
                val = evaluate(w, h, c, r)
                if val > 0:
                    return val, c, r
        except:
            pass
        return 0, None, None

    best_val = 0
    best_result = None
    
    # Try various grid layouts
    layouts = [
        (5, 4, 1),   # 5x4 grid + 1 extra
        (4, 5, 1),
        (5, 5, -4),  # 5x5 grid, remove 4
        (6, 4, -3),
        (4, 6, -3),
        (7, 3, 0),
        (3, 7, 0),
        (6, 3, 3),
        (3, 6, 3),
    ]
    
    for n_cols, n_rows, extra in layouts:
        if time.time() > deadline - 20:
            break
        val, centers, radii = pack_with_layout(n_cols, n_rows, extra)
        if val > best_val:
            best_val = val
            best_result = (centers, radii)
    
    # Also try a more global optimization from random starts
    if time.time() < deadline - 30:
        w = 1.0
        h = 1.0
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            np.random.seed(42 + attempt)
            centers = np.column_stack([
                np.random.uniform(0.1, 0.9, N) * w,
                np.random.uniform(0.1, 0.9, N) * h
            ])
            radii = np.full(N, 0.05)
            
            def objective(x):
                c = x[:2*N].reshape(N, 2)
                r = x[2*N:]
                for i in range(N):
                    if c[i][0] - r[i] < -1e-9 or c[i][0] + r[i] > w + 1e-9:
                        return 1e6
                    if c[i][1] - r[i] < -1e-9 or c[i][1] + r[i] > h + 1e-9:
                        return 1e6
                for i in range(N):
                    for j in range(i+1, N):
                        dx = c[i][0] - c[j][0]
                        dy = c[i][1] - c[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-9:
                            return 1e6 + (r[i] + r[j] - dist)
                return -np.sum(r)
            
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
            for i in range(N):
                bounds.append((0, 0.5))
            
            x0 = np.concatenate([centers.flatten(), radii])
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-12})
                if res.fun < 1e5:
                    c = res.x[:2*N].reshape(N, 2)
                    r = res.x[2*N:]
                    val = evaluate(w, h, c, r)
                    if val > best_val:
                        best_val = val
                        best_result = (c, r)
            except:
                pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    centers, radii = best_result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [centers[i][0], centers[i][1], radii[i]]
    return result
