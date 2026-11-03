import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # ---- Helpers ----
    def check_valid(x, y, r, w, h, tol=1e-5):
        if w <= 0 or h <= 0:
            return False
        if np.any(x < r - tol) or np.any(w - x < r - tol) or \
           np.any(y < r - tol) or np.any(h - y < r - tol):
            return False
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def total_radius(x, y, r, w, h):
        return float(np.sum(r))

    # ---- Initialization: hexagonal cluster ----
    def init_hex(w, h):
        # Generate hex grid points, select N closest to center
        pts = []
        for j in range(8):
            for i in range(8):
                x = i * np.sqrt(3) + (j % 2) * np.sqrt(3)/2
                y = j * 1.5
                pts.append((x, y))
        pts = np.array(pts)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        d = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
        order = np.argsort(d)
        pts = pts[order[:N]].copy()
        pts -= pts.mean(axis=0)
        # Scale to fit
        px, py = pts[:,0], pts[:,1]
        sx = w * 0.7 / (max(px.max()-px.min(), 1e-9))
        sy = h * 0.7 / (max(py.max()-py.min(), 1e-9))
        pts[:,0] = (pts[:,0] - (px.max()+px.min())/2) * sx + w/2
        pts[:,1] = (pts[:,1] - (py.max()+py.min())/2) * sy + h/2
        r = 0.05 * np.ones(N)
        return pts, r

    # ---- Force-directed relaxation ----
    def force_relax(x, y, r, w, h, iters=5000, dt=0.01):
        x = x.copy(); y = y.copy(); r = r.copy()
        # Try to grow radii uniformly
        best_r = r.copy()
        best_x = x.copy()
        best_y = y.copy()
        best_sum = np.sum(r)
        
        for it in range(iters):
            if time.time() > deadline:
                break
            # Compute forces
            fx = np.zeros(N); fy = np.zeros(N)
            # Overlap repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j]-x[i]; dy = y[j]-y[i]
                    d = np.hypot(dx, dy)
                    if d < 1e-10:
                        dx, dy = 0.01, 0.01
                        d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        f = 1.0 / max(d, 1e-6)
                        fx[i] -= f * dx/d
                        fy[i] -= f * dy/d
                        fx[j] += f * dx/d
                        fy[j] += f * dy/d
            # Boundary attraction (keep centers away from walls by r)
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 5.0 * (r[i] - x[i])
                elif x[i] > w - r[i]:
                    fx[i] += 5.0 * (w - r[i] - x[i])
                if y[i] < r[i]:
                    fy[i] += 5.0 * (r[i] - y[i])
                elif y[i] > h - r[i]:
                    fy[i] += 5.0 * (h - r[i] - y[i])
            # Update
            x += dt * fx
            y += dt * fy
            # Project to bounds
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
            
            # Try to grow radii
            if it % 50 == 0:
                # Max possible radius for each circle
                for i in range(N):
                    min_dist = np.inf
                    for j in range(N):
                        if i != j:
                            min_dist = min(min_dist, np.hypot(x[i]-x[j], y[i]-y[j]) - r[j])
                    min_dist = min(min_dist, x[i], w-x[i], y[i], h-y[i])
                    r[i] = min(min_dist, r[i] * 1.02, 1.0)
                if np.sum(r) > best_sum:
                    best_sum = np.sum(r)
                    best_r = r.copy()
                    best_x = x.copy()
                    best_y = y.copy()
        
        return best_x, best_y, best_r, best_sum

    # ---- Gradient ascent on radii (fixed centers) ----
    def grow_radii(x, y, r, w, h, iters=2000):
        r = r.copy()
        for it in range(iters):
            if time.time() > deadline:
                break
            for i in range(N):
                min_dist = np.inf
                for j in range(N):
                    if i != j:
                        min_dist = min(min_dist, np.hypot(x[i]-x[j], y[i]-y[j]) - r[j])
                min_dist = min(min_dist, x[i], w-x[i], y[i], h-y[i])
                # Grow toward min_dist
                target = min_dist
                if r[i] < target:
                    r[i] += 0.1 * (target - r[i])
                else:
                    r[i] = target
        return r

    # ---- Optimize aspect ratio for a fixed configuration ----
    def optimize_aspect(x, y, r):
        # Try different w values
        best_score = -1
        best_w = 1.0
        for w in np.linspace(0.5, 1.5, 21):
            h = 2.0 - w
            # Check if circles fit
            valid = True
            for i in range(N):
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                    valid = False
                    break
            if not valid:
                continue
            # Check pairwise
            for i in range(N):
                for j in range(i+1, N):
                    if np.hypot(x[i]-x[j], y[i]-y[j]) < r[i]+r[j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_w = w
        return best_w, best_score

    # ---- Main search ----
    best_x = None
    best_y = None
    best_r = None
    best_w = 1.0
    best_h = 1.0
    best_score = 0.0

    # Multiple aspect ratio starts
    aspect_ratios = [1.0, 1.5, 0.7, 2.0, 0.5, 1.3, 0.8]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            continue
        
        # Initialize
        x, y = init_hex(w, h)
        r = 0.03 * np.ones(N)
        
        # Force relaxation
        x, y, r, s = force_relax(x, y, r, w, h, iters=3000)
        
        # Grow radii
        r = grow_radii(x, y, r, w, h, iters=1000)
        
        # Validate
        if check_valid(x, y, r, w, h, tol=1e-5):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                best_w = w
                best_h = h
        
        # Try to optimize aspect
        w2, s2 = optimize_aspect(x, y, r)
        if s2 > best_score:
            h2 = 2.0 - w2
            if check_valid(x, y, r, w2, h2, tol=1e-5):
                best_score = s2
                best_w = w2
                best_h = h2
    
    # If no valid solution, fallback
    if best_x is None:
        # Small circles in grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.4 / side
                    cells.append([x, y, r])
        result = np.array(cells[:N])
        return result

    # Final result
    result = np.column_stack((best_x, best_y, best_r))
    return result
