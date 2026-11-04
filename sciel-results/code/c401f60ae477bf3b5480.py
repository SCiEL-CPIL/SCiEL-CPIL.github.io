import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Centers of the 11 hexagons
    # 1 center at (0,0)
    # 6 at distance sqrt(3) at angles 0, 60, ... 300
    # 4 at distance 3 at angles 30, 90, 150, 210 (arbitrary choice of 4 out of 6 gaps)
    # Note: The choice of 4 gaps might not be optimal if symmetry is broken, 
    # but optimization will handle the rotations. 
    # Actually, for 11 hexagons, the standard "flower" minus 2 is a common dense packing.
    
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    # Place 4 hexagons in the gaps. 
    # To maintain some symmetry or allow optimization, we pick 4 specific ones.
    # The optimization will rotate them to minimize the bounding box.
    for k in range(4):
        angle = np.deg2rad(30 + k * 60) # 30, 90, 150, 210
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        # Note: If theta_out is 0, normals are at 30, 90, ...
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(dots))
        
        # For a regular hexagon with side S, apothem = S * sqrt(3) / 2
        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)
    
    # Initial guess: 
    # Outer rotation 0.
    # Inner rotations: 
    # Try to align faces radially.
    # For a hex at angle alpha, radial normal is alpha.
    # Base hex face normals are at 30, 90...
    # To align face normal to alpha, rotate by alpha - 30.
    x0 = np.zeros(n_hex + 1)
    x0[0] = 0.0
    for i in range(n_hex):
        cx, cy = centers[i]
        if cx == 0 and cy == 0:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(cy, cx)
            # Align face normal (30 deg in base) to alpha
            x0[i+1] = alpha - np.deg2rad(30)
            
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start Powell optimization
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # Try all vertices aligned
            x_start = np.zeros(n_hex + 1)
        else:
            # Random perturbations
            x_start = x0 + np.random.uniform(-0.1, 0.1, n_hex + 1)
            # Ensure theta_out is in a reasonable range
            x_start[0] = np.random.uniform(0, np.pi/3)
            
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to analytical solution
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
            
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    side = 4.0 + 1e-7
    return inner, (0.0, 0.0), side, 30.0
