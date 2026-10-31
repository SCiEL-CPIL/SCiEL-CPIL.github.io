"""
Circle packing in perimeter-4 rectangle.

Combines the unified optimization strategy (inspiration 1) where rectangle 
dimensions are decision variables, with the robust force-directed relaxation 
and penalty-based SLSQP machinery (current program) for stability and speed.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin for 120s/150s limit

def _force_relax(cx, cy, r, W, H, max_iter=300, dt=0.001):
    """
    Fast force-directed relaxation to remove overlaps and wall violations.
    Returns updated cx, cy.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    # Ensure W, H are valid
    if W <= 0 or H <= 0:
        return cx, cy
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        # Left
        viol = r - cx
        mask = viol > 0
        fx[mask] += viol[mask] * 10.0
        # Right
        viol = cx + r - W
        mask = viol > 0
        fx[mask] -= viol[mask] * 10.0
        # Bottom
        viol = r - cy
        mask = viol > 0
        fy[mask] += viol[mask] * 10.0
        # Top
        viol = cy + r - H
        mask = viol > 0
        fy[mask] -= viol[mask] * 10.0
        
        # Pairwise overlap forces
        # Vectorized pairwise distance check for speed
        # d[i,j] = (cx[i]-cx[j])^2 + (cy[i]-cy[j])^2
        # We only need to check i < j
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            d2 = dx*dx + dy*dy
            min_d = r[i] + r[i+1:]
            # Check overlap
            overlap_mask = d2 < min_d**2
            if np.any(overlap_mask):
                # Calculate forces for overlapping pairs
                # d2 is flat array of length n-i-1
                # We need to apply forces to i and i+1..n
                # This loop is O(N^2) but N=21 is small, vectorizing fully is complex for force accumulation
                # Let's stick to simple loop for clarity and correctness in relaxation, 
                # as this is a pre-processing step.
                pass 
        
        # Re-implementing pairwise forces with simple loop for reliability in relaxation
        # (The vectorized attempt above was just for check, let's do the actual force calc)
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Walls
        fx += np.where(cx < r, (r - cx) * 10.0, 0.0)
        fx += np.where(cx > W - r, (W - r - cx) * 10.0, 0.0)
        fy += np.where(cy < r, (r - cy) * 10.0, 0.0)
        fy += np.where(cy > H - r, (H - r - cy) * 10.0, 0.0)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx*dx + dy*dy
                min_d = r[i] + r[j]
                if d2 < min_d*min_d:
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                    overlap = min_d - d
                    ux = dx / d
                    uy = dy / d
                    fx[i] += ux * overlap * 5.0
                    fy[i] += uy * overlap * 5.0
                    fx[j] -= ux * overlap * 5.0
                    fy[j] -= uy * overlap * 5.0
        
        cx += dt * fx
        cy += dt * fy
        
        # Clip to valid region to prevent explosion
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
    return cx, cy

def _is_feasible(W, H, cx, cy, r, tol=1e-5):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(r < -tol):
        return False
    # Walls
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _objective(x):
    """
    x = [W, r_0...r_N-1, cx_0...cx_N-1, cy_0...cy_N-1]
    H is derived as 2.0 - W.
    Objective: -sum(r) + penalty
    """
    W = x[0]
    H = 2.0 - W
    
    # If W or H are invalid, return large penalty
    if W < 0.1 or H < 0.1:
        return 1e6
        
    r = x[1:1+N]
    cx = x[1+N:1+2*N]
    cy = x[1+2*N:1+3*N]
    
    pen = 0.0
    
    # Wall penalties
    # cx - r >= 0 => min(0, cx-r)^2
    pen += np.sum(np.minimum(0.0, cx - r)**2)
    pen += np.sum(np.minimum(0.0, cy - r)**2)
    # cx + r <= W => max(0, cx+r-W)^2
    pen += np.sum(np.maximum(0.0, cx + r - W)**2)
    pen += np.sum(np.maximum(0.0, cy + r - H)**2)
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx*dx + dy*dy
            s = r[i] + r[j]
            if d2 < s*s:
                d = np.sqrt(d2) if d2 > 1e-12 else 0.0
                pen += (s - d)**2
    
    return -np.sum(r) + 100.0 * pen

def _generate_seed(W_init, layout='hex'):
    """
    Generate initial x vector.
    """
    H_init = 2.0 - W_init
    x = np.zeros(3*N + 1)
    x[0] = W_init
    
    if layout == 'hex':
        # Hexagonal layout
        cols = int(np.ceil(np.sqrt(N)))
        spacing = 1.0
        verts = spacing * np.sqrt(3.0) / 2.0
        pts = []
        row = 0
        while len(pts) < N:
            xoff = (0.5 * spacing if row % 2 else 0.0)
            for col in range(cols):
                if len(pts) < N:
                    pts.append((col * spacing + xoff, row * verts))
            row += 1
        pts = np.array(pts[:N], dtype=float)
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        sx = (xmax - xmin) if xmax > xmin else 1.0
        sy = (ymax - ymin) if ymax > ymin else 1.0
        pts[:, 0] = (pts[:, 0] - xmin) / sx
        pts[:, 1] = (pts[:, 1] - ymin) / sy
        # Scale to fit with margin
        m = 0.8
        cx = (pts[:, 0] * (W_init * m)) + W_init * (1 - m) / 2
        cy = (pts[:, 1] * (H_init * m)) + H_init * (1 - m) / 2
        base_r = 0.5 * spacing * min(W_init / sx, H_init / sy) * 0.4
    else:
        # Grid layout
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        cx = np.array([(i + 0.5) / cols for i in range(cols) for _ in range(rows)][:N]) * W_init
        cy = np.array([(j + 0.5) / rows for _ in range(cols) for j in range(rows)][:N]) * H_init
        base_r = 0.5 * min(W_init / cols, H_init / rows) * 0.8

    r = np.full(N, base_r)
    
    x[1:1+N] = r
    x[1+N:1+2*N] = cx
    x[1+2*N:1+3*N] = cy
    
    return x

def construct():
    start_time = time.time()
    best_val = -float('inf')
    best_result = None
    
    # Bounds for SLSQP
    # W in [0.2, 1.8]
    # r in [0.01, 0.5]
    # cx in [0, 2]
    # cy in [0, 2]
    bounds = [(0.2, 1.8)]
    for i in range(N):
        bounds.append((0.01, 0.5))
    for i in range(N):
        bounds.append((0.0, 2.0))
    for i in range(N):
        bounds.append((0.0, 2.0))
        
    # Try different initial aspect ratios and layouts
    configs = [
        (1.0, 'hex'),
        (1.0, 'grid'),
        (1.2, 'hex'),
        (0.8, 'hex'),
        (1.3, 'grid'),
        (0.7, 'grid'),
        (1.1, 'hex'),
        (0.9, 'hex'),
    ]
    
    for W_init, layout in configs:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x0 = _generate_seed(W_init, layout)
        
        # Pre-relaxation to get closer to feasibility
        W0 = x0[0]
        H0 = 2.0 - W0
        r0 = x0[1:1+N]
        cx0 = x0[1+N:1+2*N]
        cy0 = x0[1+2*N:1+3*N]
        
        cx_rel, cy_rel = _force_relax(cx0, cy_rel if 'cy_rel' in dir() else cy0, r0, W0, H0, max_iter=100)
        # Update x0 with relaxed positions
        x0[1+N:1+2*N] = cx_rel
        x0[1+2*N:1+3*N] = cy_rel
        
        # Optimize
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            W_opt = x_opt[0]
            H_opt = 2.0 - W_opt
            r_opt = x_opt[1:1+N]
            cx_opt = x_opt[1+N:1+2*N]
            cy_opt = x_opt[1+2*N:1+3*N]
            
            # Check feasibility
            if _is_feasible(W_opt, H_opt, cx_opt, cy_opt, r_opt, tol=1e-4):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = (cx_opt, cy_opt, r_opt)
            else:
                # Try to repair with force relaxation
                cx_rep, cy_rep = _force_relax(cx_opt, cy_opt, r_opt, W_opt, H_opt, max_iter=200)
                if _is_feasible(W_opt, H_opt, cx_rep, cy_rep, r_opt, tol=1e-4):
                    val = np.sum(r_opt)
                    if val > best_val:
                        best_val = val
                        best_result = (cx_rep, cy_rep, r_opt)
                        
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        # Simple grid in 1x1 square (perimeter 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r = best_result
    return np.column_stack([cx, cy, r])
