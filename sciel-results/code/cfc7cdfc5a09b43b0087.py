"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = np.deg2rad(30) + angle_rad + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _support_function(vertices, direction):
    """Max of dot(v, direction) over vertices."""
    return np.max(np.dot(vertices, direction))

def _compute_min_outer_side(inner_hexes, outer_center, outer_angle_rad):
    """
    Compute the minimum outer side length such that all inner hexagons fit inside
    the outer hexagon (axis-aligned in the outer frame).
    
    inner_hexes: list of (x, y, angle_rad)
    outer_center: (cx, cy)
    outer_angle_rad: rotation of outer hexagon
    
    Returns the minimum side length s.
    """
    ocx, ocy = outer_center
    
    # The outer hexagon in its local frame has vertices at angles 0, 60, ..., 300 degrees
    # relative to its own center. In the global frame, these are rotated by outer_angle_rad.
    # A regular hexagon with side s centered at origin has support function in direction u:
    # h(u) = s * max_{k=0..5} cos(angle_k - theta_u) where angle_k are vertex angles.
    # Actually for a regular hexagon, the support function in direction making angle theta with 
    # the hexagon's axis is: s * max(cos(theta - k*pi/3) for k=0..5)
    # But more precisely, the hexagon is the intersection of 6 half-planes.
    # The distance from center to each face is s * cos(pi/6) = s * sqrt(3)/2.
    # The 6 face normals are at angles: outer_angle_rad + k*pi/3 for k=0..5.
    # Wait, let me think again.
    
    # A regular hexagon with side length s, centered at origin, with one vertex pointing in direction 0:
    # Vertices at s * exp(i * k*pi/3) for k=0..5.
    # The 6 supporting lines (faces) have normals at angles (k+0.5)*pi/3 for k=0..5.
    # Distance from center to each face = s * cos(pi/6) = s * sqrt(3)/2.
    
    # So the outer hexagon (rotated by outer_angle_rad) satisfies:
    # For each of 6 directions n_k = exp(i*(outer_angle_rad + (k+0.5)*pi/3)), k=0..5:
    #   max_{v in inner_i} dot(v - outer_center, n_k) <= s * sqrt(3)/2
    #   AND max_{v in inner_i} dot(v - outer_center, -n_k) <= s * sqrt(3)/2
    # (both sides of each face)
    
    # So s >= (2/sqrt(3)) * max over all 12 directions and all inner hexagons of support.
    
    # 12 directions: outer_angle_rad + (k+0.5)*pi/3 and outer_angle_rad + (k+0.5)*pi/3 + pi for k=0..5
    directions = []
    for k in range(6):
        base_angle = outer_angle_rad + (k + 0.5) * np.pi / 3
        directions.append(np.array([np.cos(base_angle), np.sin(base_angle)]))
        directions.append(np.array([-np.cos(base_angle), -np.sin(base_angle)]))
    
    max_support = 0.0
    for (ix, iy, iangle) in inner_hexes:
        vx, vy = _hex_vertices(ix, iy, 1.0, iangle)
        # Translate to outer center frame
        vx = vx - ocx
        vy = vy - ocy
        for d in directions:
            s_val = np.max(vx * d[0] + vy * d[1])
            if s_val > max_support:
                max_support = s_val
    
    return 2.0 * max_support / np.sqrt(3.0)

def _objective(params, outer_center, outer_angle_rad):
    """
    params: flat array of length 3*N, each triple is (x, y, angle_rad) for each inner hex.
    Returns the computed outer side length.
    """
    inner_hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, a))
    return _compute_min_outer_side(inner_hexes, outer_center, outer_angle_rad)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: return by 120s, evaluator kills at 150s
    
    np.random.seed(42)
    
    # Initial configuration: place hexagons in a hexagonal grid pattern
    # Hexagonal grid spacing for unit hexagons:
    # Horizontal spacing: 2 * sqrt(3) * 0.5 = sqrt(3) for adjacent columns? 
    # Actually, for regular hexagons of side 1:
    # Width = 2, height = sqrt(3)
    # In a hex grid, horizontal distance between centers in same row = 2
    # Vertical distance between rows = sqrt(3)
    # Offset every other row by 1
    
    # Let's use a compact hexagonal arrangement for 12 hexagons.
    # 12 = 1 + 3 + 4 + 3 + 1? No, that's 12 but not standard.
    # Let's try: rows of 3, 4, 3, 2? Or 4, 4, 4?
    # 4 rows of 3: 12 hexagons.
    
    dx = 2.0  # horizontal spacing
    dy = np.sqrt(3)  # vertical spacing
    
    # Start with a 4x3 grid arrangement
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * 1.0
            y = row * dy
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (max(xs) + min(xs)) / 2.0
    cy0 = (max(ys) + min(ys)) / 2.0
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    # Initial outer parameters
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Convert to flat params
    params0 = np.array([v for p in positions for v in p])
    
    # Initial evaluation
    best_side = _objective(params0, outer_center, outer_angle)
    best_params = params0.copy()
    
    # Use simulated annealing / random search with local optimization
    # Strategy: 
    # 1. Try different outer angles
    # 2. For each, optimize inner positions
    # 3. Keep best
    
    best_overall = best_side
    best_overall_params = best_params.copy()
    best_overall_outer_center = outer_center
    best_overall_outer_angle = outer_angle
    
    # Try a few outer angles
    for outer_angle_try in [0.0, np.pi/6, np.pi/3, np.pi/12, -np.pi/12, np.pi/4]:
        if time.time() > deadline - 10.0:
            break
        
        # Optimize inner positions for this outer angle
        # Use scipy minimize with Nelder-Mead or L-BFGS-B
        # The objective is non-smooth due to max, but we can try
        
        # Start from best known or from initial
        x0 = best_overall_params.copy()
        
        try:
            result = minimize(
                lambda p: _objective(p, outer_center, outer_angle_try),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_overall:
                best_overall = result.fun
                best_overall_params = result.x.copy()
                best_overall_outer_angle = outer_angle_try
                best_overall_outer_center = outer_center
        except Exception:
            pass
        
        # Also try with a random perturbation
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            x0 = best_overall_params + np.random.normal(0, 0.5, size=3*N)
            try:
                result = minimize(
                    lambda p: _objective(p, outer_center, outer_angle_try),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if result.fun < best_overall:
                    best_overall = result.fun
                    best_overall_params = result.x.copy()
                    best_overall_outer_angle = outer_angle_try
                    best_overall_outer_center = outer_center
            except Exception:
                pass
    
    # Also try optimizing the outer center and angle jointly with inner positions
    # This is a 3N + 3 dimensional problem
    def _full_objective(params_full):
        """
        params_full: [x1,y1,a1, ..., xN,yN,aN, cx, cy, ca]
        """
        inner_params = params_full[:3*N]
        ocx = params_full[3*N]
        ocy = params_full[3*N+1]
        oca = params_full[3*N+2]
        return _objective(inner_params, (ocx, ocy), oca)
    
    # Start from best
    x0_full = np.concatenate([best_overall_params, [best_overall_outer_center[0], 
                                                      best_overall_outer_center[1], 
                                                      best_overall_outer_angle]])
    
    try:
        result = minimize(
            _full_objective,
            x0_full,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
        )
        if result.fun < best_overall:
            best_overall = result.fun
            best_overall_params = result.x[:3*N].copy()
            best_overall_outer_center = (result.x[3*N], result.x[3*N+1])
            best_overall_outer_angle = result.x[3*N+2]
    except Exception:
        pass
    
    # Additional random restarts if time permits
    while time.time() < deadline - 15.0:
        # Random perturbation of best solution
        x0 = best_overall_params + np.random.normal(0, 1.0, size=3*N)
        oc = (best_overall_outer_center[0] + np.random.normal(0, 0.5),
              best_overall_outer_center[1] + np.random.normal(0, 0.5))
        oa = best_overall_outer_angle + np.random.normal(0, 0.3)
        
        try:
            result = minimize(
                lambda p: _objective(p, oc, oa),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_overall:
                best_overall = result.fun
                best_overall_params = result.x.copy()
                best_overall_outer_center = oc
                best_overall_outer_angle = oa
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        x0_full = np.concatenate([best_overall_params, [best_overall_outer_center[0], 
                                                          best_overall_outer_center[1], 
                                                          best_overall_outer_angle]])
        try:
            result = minimize(
                _full_objective,
                x0_full,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_overall:
                best_overall = result.fun
                best_overall_params = result.x[:3*N].copy()
                best_overall_outer_center = (result.x[3*N], result.x[3*N+1])
                best_overall_outer_angle = result.x[3*N+2]
        except Exception:
            pass
    
    # Convert back to required format
    inner = []
    for i in range(N):
        x = best_overall_params[3*i]
        y = best_overall_params[3*i+1]
        a_rad = best_overall_params[3*i+2]
        a_deg = np.rad2deg(a_rad)
        inner.append([x, y, a_deg])
    
    return inner, best_overall_outer_center, best_overall, best_overall_outer_angle
# EVOLVE-BLOCK-END
