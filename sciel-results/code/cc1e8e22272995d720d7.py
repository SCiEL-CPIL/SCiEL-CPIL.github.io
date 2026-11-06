"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    We use the row configuration [7, 6, 7, 6, 7, 6, 7, 4] which sums to 50.
    We scale the lattice to fit the square [-0.5, 0.5]^2 as tightly as possible.
    """
    rows = [7, 6, 7, 6, 7, 6, 7, 4]
    
    # Generate ideal hex positions
    # Standard hex spacing s. Vertical distance h = s * sqrt(3)/2.
    # Horizontal offset for alternating rows is s/2.
    # Let's build the points in a local coordinate system first.
    pts_local = []
    h = np.sqrt(3.0) / 2.0
    for i, n in enumerate(rows):
        y = i * h
        # Offset every other row
        offset = 0.5 if (i % 2 != 0) else 0.0
        # Center the row horizontally
        # x positions: 0, 1, ..., n-1
        xs = np.arange(n) + offset
        # Center this row relative to the widest row (width 7 -> range 6)
        # Actually, let's just center each row independently first, then adjust global center
        center_x = (n - 1) / 2.0
        xs = xs - center_x
        for x in xs:
            pts_local.append([x, y])
    
    pts_local = np.array(pts_local)
    
    # Now scale and translate pts_local to fit in [-0.5, 0.5]
    # Find bounds of local points
    min_x, min_y = pts_local.min(axis=0)
    max_x, max_y = pts_local.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to fit width and height into 1.0 x 1.0 square.
    # The limiting factor is the larger aspect ratio compared to 1:1.
    # scale = min(1.0 / width, 1.0 / height)
    # However, we also need to ensure the points are strictly inside if we want to be safe, 
    # but the optimizer will handle the boundary. 
    # Scaling to exactly fit might place points on the boundary, which is fine for initialization.
    scale = min(1.0 / width, 1.0 / height)
    
    # Scale points
    pts_scaled = pts_local * scale
    
    # Center the scaled points
    center_local = (pts_local.mean(axis=0))
    center_scaled = center_local * scale
    pts_centered = pts_scaled - center_scaled
    
    return pts_centered

def construct():
    import time
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_pts = _make_hex_initial()
    best_score = _score(best_pts)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Use Powell's method which is often faster and more robust than Nelder-Mead 
    # for this type of problem in 100D.
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Powell",
                       options={'maxiter': 500, 'ftol': 1e-8, 'xtol': 1e-8, 'disp': False})
        
        if time.time() > deadline - 5.0:
            return best_pts.reshape(N, 2)
            
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass

    # If time permits, try Nelder-Mead as a second phase for finer tuning
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                            options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000, 'adaptive': True})
            
            if time.time() > deadline - 5.0:
                return best_pts.reshape(N, 2)
                
            new_pts = res2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    # If time still permits, try a different initialization (jittered)
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        jittered = _make_hex_initial() + np.random.normal(0, 0.005, (N, 2))
        try:
            res3 = minimize(_neg_score, jittered.flatten(), method="Powell",
                            options={'maxiter': 300, 'ftol': 1e-8, 'xtol': 1e-8, 'disp': False})
            
            if time.time() > deadline - 5.0:
                return best_pts.reshape(N, 2)
                
            new_pts = res3.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
