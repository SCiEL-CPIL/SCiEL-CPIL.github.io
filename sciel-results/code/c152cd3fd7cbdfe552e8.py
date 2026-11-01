import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
import sys

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _delaunay_init(w0, h0, n_circles, jitter=0.05, seed=None):
    """
    Generate initial non-overlapping circles using Delaunay triangulation.
    Places circles at the in-centers of Delaunay triangles with in-radii.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Create a grid of points slightly larger than N to ensure enough triangles
    side = int(np.ceil(np.sqrt(n_circles * 1.5))) 
    xs = np.linspace(jitter, w0 - jitter, side)
    ys = np.linspace(jitter, h0 - jitter, side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Add jitter to avoid degenerate Delaunay
    points += rng.uniform(-jitter, jitter, points.shape)
    
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback to simple grid if Delaunay fails
        return _grid_init(w0, h0, n_circles)

    # Get centroids and inradii of triangles
    # Note: We want circles that don't overlap. Using triangle in-centers and in-radii is a standard way to get a packing.
    # However, adjacent triangles' incircles might still overlap if the triangulation is not equilateral enough.
    # To be safe, we can scale down the radii by a factor.
    
    tri_verts = points[tri.simplices]
    # Calculate incenter and inradius for each triangle
    incents = []
    inradii = []
    
    for v in tri_verts:
        a = v[0]
        b = v[1]
        c = v[2]
        
        # Side lengths
        l_ab = np.linalg.norm(b - a)
        l_bc = np.linalg.norm(c - b)
        l_ca = np.linalg.norm(a - c)
        
        # Incenter
        # Formula: (a*A + b*B + c*C) / (a+b+c) where a,b,c are side lengths opposite to vertices A,B,C
        # Wait, standard formula: Incenter = (l_a * A + l_b * B + l_c * C) / (l_a + l_b + l_c)
        # l_a is length of side opposite A (BC), etc.
        l_a = l_bc
        l_b = l_ca
        l_c = l_ab
        
        s = l_a + l_b + l_c
        if s < 1e-9:
            continue
            
        incenter = (l_a * a + l_b * b + l_c * c) / s
        
        # Inradius
        # Area = sqrt(s(s-a)(s-b)(s-c))
        s_val = s / 2.0
        area_sq = s_val * (s_val - l_a) * (s_val - l_b) * (s_val - l_c)
        if area_sq < 0:
            continue
        area = np.sqrt(area_sq)
        inradius = area / s_val
        
        incents.append(incenter)
        inradii.append(inradius)
        
    if len(incents) == 0:
        return _grid_init(w0, h0, n_circles)

    incents = np.array(incents)
    inradii = np.array(inradii)
    
    # Filter out circles that are too close to walls or too small
    mask = (inradii > 0.01) 
    mask &= (incents[:, 0] - inradii > 0)
    mask &= (incents[:, 0] + inradii < w0)
    mask &= (incents[:, 1] - inradii > 0)
    mask &= (incents[:, 1] + inradii < h0)
    
    incents = incents[mask]
    inradii = inradii[mask]
    
    # Sort by radius (largest first) and pick top N
    if len(incents) < n_circles:
        return _grid_init(w0, h0, n_circles)
        
    order = np.argsort(-inradii)
    incents = incents[order]
    inradii = inradii[order]
    
    # Select first N
    incents = incents[:n_circles]
    inradii = inradii[:n_circles]
    
    # Ensure no overlaps by shrinking radii if necessary (greedy check)
    # Since they come from Delaunay, they *should* be non-overlapping incircles of adjacent triangles?
    # Actually, incircles of adjacent triangles are disjoint.
    # Incircles of non-adjacent triangles are also disjoint.
    # So this should be a valid packing.
    
    cx = incents[:, 0]
    cy = incents[:, 1]
    r = inradii * 0.95 # Safety factor
    
    return cx, cy, r

def _grid_init(w0, h0, n_circles):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n_circles)))
    cx = np.zeros(n_circles)
    cy = np.zeros(n_circles)
    r = np.zeros(n_circles)
    for i in range(n_circles):
        row, col = divmod(i, side)
        cx[i] = (col + 0.5) / side * w0
        cy[i] = (row + 0.5) / side * h0
        r[i] = min(w0 / side, h0 / side) * 0.45
    return cx, cy, r

def _force_relax(cx, cy, r, w, h, max_iter=500, tol=1e-6):
    """Force-directed relaxation to resolve overlaps and wall violations."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    dt = 0.01
    for _ in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces
        for i in range(N):
            if cx[i] < r[i]: fx[i] += 100 * (r[i] - cx[i])
            if cx[i] > w - r[i]: fx[i] -= 100 * (cx[i] - (w - r[i]))
            if cy[i] < r[i]: fy[i] += 100 * (r[i] - cy[i])
            if cy[i] > h - r[i]: fy[i] -= 100 * (cy[i] - (h - r[i]))
            
        # Pair forces
        max_overlap = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    overlap = r_sum - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-9:
                        ux = dx / d
                        uy = dy / d
                        f = 10 * overlap
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
                    else:
                        # Jitter if coincident
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)
                        
        if max_overlap < tol:
            break
            
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to walls
        for i in range(N):
            cx[i] = np.clip(cx[i], r[i], w - r[i])
            cy[i] = np.clip(cy[i], r[i], h - r[i])
            
    return cx, cy, r

def _slsqp_optimize(cx, cy, r, w, h, time_budget=30.0):
    """Run SLSQP to maximize sum of radii."""
    
    def objective(z):
        return -np.sum(z[2::3][:N])
        
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w)) # x
        bounds.append((0, h)) # y
        bounds.append((0.001, 0.5)) # r
    bounds.append((0.2, 1.8)) # w (h = 2-w)
    
    # Constraints
    constraints = []
    
    # Wall constraints
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
        # w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[-1] - z[3*i] - z[3*i+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
        # h - y_i - r_i >= 0  =>  (2 - w) - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: (2.0 - z[-1]) - z[3*i+1] - z[3*i+2]})
        
    # Pair constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda z, i=i, j=j: (z[3*i] - z[3*j])**2 + (z[3*i+1] - z[3*j+1])**2 - (z[3*i+2] + z[3*j+2])**2
            })
            
    # Initial guess
    z0 = np.zeros(3*N + 1)
    z0[0::3][:N] = cx
    z0[1::3][:N] = cy
    z0[2::3][:N] = r
    z0[-1] = w
    
    start_time = time.time()
    maxiter = 200
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success and res.x is not None:
            x_opt = res.x[0::3][:N]
            y_opt = res.x[1::3][:N]
            r_opt = res.x[2::3][:N]
            w_opt = res.x[-1]
            h_opt = 2.0 - w_opt
            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                return x_opt, y_opt, r_opt, w_opt, h_opt
    except Exception:
        pass
        
    return cx, cy, r, w, h

def _expand_radii(cx, cy, r, w, h, max_rounds=10, time_budget=20.0):
    """Greedy expansion of radii."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    start_time = time.time()
    
    for _ in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        improved = False
        for i in range(N):
            # Calculate max possible radius for i
            r_max = min(cx[i], w - cx[i], cy[i], h - cy[i])
            for j in range(N):
                if i == j: continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                r_max = min(r_max, r_allowed)
                
            if r_max > r[i] + 1e-5:
                # Binary search for best radius
                lo, hi = r[i], r_max
                for _ in range(20):
                    mid = (lo + hi) / 2
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-6:
                    r[i] = lo
                    improved = True
        if not improved:
            break
            
    return cx, cy, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_val = 0.0
    best_result = None
    
    # Different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.67, 1.5, 0.5, 2.0, 0.75, 1.33]
    
    for idx, w0 in enumerate(aspect_ratios):
        if time.time() > deadline - 30: # Leave 30s for final polishing
            break
        h0 = 2.0 - w0
        if h0 < 0.2: continue
        
        # Initialize
        cx, cy, r = _delaunay_init(w0, h0, N, seed=42 + idx)
        
        # Relax
        cx, cy, r = _force_relax(cx, cy, r, w0, h0, max_iter=200)
        
        # Optimize
        if time.time() > deadline - 40: continue
        cx, cy, r, w_opt, h_opt = _slsqp_optimize(cx, cy, r, w0, h0, time_budget=20.0)
        
        # Expand
        if time.time() > deadline - 15: continue
        cx, cy, r = _expand_radii(cx, cy, r, w_opt, h_opt, max_rounds=5, time_budget=10.0)
        
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_result = (cx.copy(), cy.copy(), r.copy(), w_opt, h_opt)
            
    if best_result is None:
        # Fallback
        cx, cy, r = _grid_init(1.0, 1.0, N)
        return np.column_stack((cx, cy, r))
        
    cx, cy, r, w, h = best_result
    
    # Final polish: SLSQP again on the best result
    if time.time() < deadline - 5:
        cx, cy, r, w, h = _slsqp_optimize(cx, cy, r, w, h, time_budget=15.0)
        if _is_valid(cx, cy, r, w, h):
            pass
        else:
            # Revert if invalid
            cx, cy, r, w, h = best_result
            
    # Final expansion
    if time.time() < deadline - 2:
        cx, cy, r = _expand_radii(cx, cy, r, w, h, max_rounds=5, time_budget=5.0)
        
    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(cx, cy, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((cx, cy, r))

if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
