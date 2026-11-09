"""Packing 12 unit hexagons in minimal regular hexagon using L-BFGS-B optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3.0)
INRADIUS = SQRT3 / 2.0  # inradius of unit hexagon

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side length."""
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def hex_support(cx, cy, angle_deg, dx, dy):
    """Support function of unit hexagon: max of dot(v, (dx,dy)) over vertices."""
    angle_rad = np.deg2rad(angle_deg)
    # Support of regular hexagon in direction (dx,dy):
    # = inradius * (1 + 2 * max_k |cos(theta - phi_k)|) where phi_k are normal directions
    # Actually simpler: just compute from the 3 half-plane constraints
    # For a regular hexagon with center (cx,cy) and rotation angle_rad:
    # The support in direction d is: cx*dx + cy*dy + inradius * (1 + 2*max(0, |cos(delta)|))
    # where delta is angle between d and nearest edge normal
    # Edge normals are at angles: angle_rad + k*pi/3 for k=0..5
    # But we can use the formula: h(d) = inradius * (|cos(a)| + |cos(a - pi/3)| + |cos(a + pi/3)|) / 1
    # Wait, let me just compute directly from vertices for correctness
    
    # Actually the support function of a regular hexagon (flat-top, side=1) in direction d:
    # h(d) = inradius * (|cos(theta)| + |cos(theta - pi/3)| + |cos(theta + pi/3)|)
    # where theta = angle of d minus rotation angle
    # This is because the hexagon is the intersection of 3 strips of width 2*inradius
    
    theta = np.arctan2(dy, dx) - angle_rad
    # Reduce to [0, pi/3)
    theta = theta % (np.pi / 3.0)
    # h = inradius * (1 + 2*|cos(theta)|) ... no that's not right either
    # Let me just use the vertex formula
    h = 0.0
    for k in range(6):
        phi = angle_rad + k * np.pi / 3.0
        vx = np.cos(phi)
        vy = np.sin(phi)
        val = vx * dx + vy * dy
        if val > h:
            h = val
    return cx * dx + cy * dy + h

def min_separation(c1, a1, c2, a2):
    """Minimum separation between two unit hexagons. Returns negative if overlapping."""
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    
    # For two convex polygons A, B, the minimum distance is:
    # min over all directions d of [dot(d, c2-c1) - h_A(d) - h_B(-d)]
    # For regular hexagons, we only need to check edge normal directions
    # Edge normals of hex1: a1 + k*pi/3, k=0..5
    # Edge normals of hex2: a2 + k*pi/3, k=0..5
    
    min_dist = float('inf')
    
    for base in [a1, a2]:
        for k in range(6):
            theta = np.deg2rad(base) + k * np.pi / 3.0
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            # h_A(d) where d = (nx, ny)
            h1 = hex_support(c1[0], c1[1], a1, nx, ny)
            # h_B(-d) where -d = (-nx, -ny)
            h2 = hex_support(c2[0], c2[1], a2, -nx, -ny)
            
            gap = dx * nx + dy * ny - h1 - h2
            if gap < min_dist:
                min_dist = gap
    
    return min_dist

def outer_hex_penalty(x, y, L):
    """Penalty for point (x,y) being outside regular hexagon centered at origin with side L."""
    inr = L * INRADIUS
    # Three half-plane constraints for flat-top hexagon
    d0x, d0y = 1.0, 0.0
    d1x, d1y = 0.5, SQRT3/2.0
    d2x, d2y = -0.5, SQRT3/2.0
    
    h0 = abs(x * d0x + y * d0y)
    h1 = abs(x * d1x + y * d1y)
    h2 = abs(x * d2x + y * d2y)
    
    pen = 0.0
    if h0 > inr:
        pen += (h0 - inr) ** 2
    if h1 > inr:
        pen += (h1 - inr) ** 2
    if h2 > inr:
        pen += (h2 - inr) ** 2
    return pen

def objective(params):
    """
    params: 3*N + 1 = [x1,y1,a1, ..., xN,yN,aN, L]
    Returns: L + penalty
    """
    L = params[-1]
    inner = params[:-1]
    
    penalty = 0.0
    
    # Check all vertices of all inner hexagons are inside outer hexagon
    for i in range(N):
        xi = inner[3*i]
        yi = inner[3*i+1]
        ai = inner[3*i+2]
        vi = hex_vertices(xi, yi, ai, 1.0)
        for vx, vy in vi:
            penalty += outer_hex_penalty(vx, vy, L)
    
    # Check non-overlap
    for i in range(N):
        for j in range(i+1, N):
            ci = np.array([inner[3*i], inner[3*i+1]])
            cj = np.array([inner[3*j], inner[3*j+1]])
            ai = inner[3*i+2]
            aj = inner[3*j+2]
            dist = min_separation(ci, ai, cj, aj)
            if dist < 0:
                penalty += dist ** 2
    
    return L + 1000.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_L = 50.0
    best_params = None
    
    def try_optimize(x0, maxiter=300):
        nonlocal best_L, best_params
        if time.time() > deadline - 10.0:
            return
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            L_cand = result.x[-1]
            inner_cand = result.x[:-1]
            # Verify
            pen = 0.0
            for i in range(N):
                xi = inner_cand[3*i]
                yi = inner_cand[3*i+1]
                ai = inner_cand[3*i+2]
                vi = hex_vertices(xi, yi, ai, 1.0)
                for vx, vy in vi:
                    pen += outer_hex_penalty(vx, vy, L_cand)
            for i in range(N):
                for j in range(i+1, N):
                    ci = np.array([inner_cand[3*i], inner_cand[3*i+1]])
                    cj = np.array([inner_cand[3*j], inner_cand[3*j+1]])
                    dist = min_separation(ci, inner_cand[3*i+2], cj, inner_cand[3*j+2])
                    if dist < -1e-6:
                        pen += dist ** 2
            if pen < 1e-4 and L_cand < best_L:
                best_L = L_cand
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Generate diverse initial guesses
    np.random.seed(123)
    initial_guesses = []
    
    # Hexagonal lattice arrangement
    # Unit hexagons in a honeycomb: centers at distance 2*inradius = sqrt(3) apart in 3 directions
    # For a cluster of 12: center + 6 around + 5 more
    # Let's try a 4-3-4-1 or similar arrangement
    
    # Arrangement 1: Two rows of 6
    params = np.zeros(3*N + 1)
    for i in range(6):
        params[3*i] = (i - 2.5) * 2.0
        params[3*i+1] = -1.0
        params[3*i+2] = 0.0
    for i in range(6):
        params[3*(i+6)] = (i - 2.5) * 2.0 + 1.0
        params[3*(i+6)+1] = 1.0
        params[3*(i+6)+2] = 0.0
    params[-1] = 7.0
    initial_guesses.append(params.copy())
    
    # Arrangement 2: Hexagonal cluster (1+6+5)
    params = np.zeros(3*N + 1)
    # Center
    params[0], params[1], params[2] = 0, 0, 0
    # 6 around
    for k in range(6):
        ang = k * np.pi / 3
        idx = 3*(k+1)
        params[idx] = 2.0 * np.cos(ang)
        params[idx+1] = 2.0 * np.sin(ang)
        params[idx+2] = 0.0
    # 5 more in a ring
    for k in range(5):
        ang = k * 2 * np.pi / 5 + np.pi/5
        idx = 3*(k+7)
        params[idx] = 3.5 * np.cos(ang)
        params[idx+1] = 3.5 * np.sin(ang)
        params[idx+2] = 0.0
    params[-1] = 6.0
    initial_guesses.append(params.copy())
    
    # Arrangement 3: 3-4-3-2 grid
    rows = [(0, 3), (1, 4), (2, 3), (3, 2)]
    idx = 0
    params = np.zeros(3*N + 1)
    for row, count in rows:
        for col in range(count):
            params[3*idx] = (col - (count-1)/2.0) * 2.1
            params[3*idx+1] = (row - 1.5) * 1.85
            params[3*idx+2] = 0.0
            idx += 1
    params[-1] = 6.0
    initial_guesses.append(params.copy())
    
    # Arrangement 4: Ring of 12
    params = np.zeros(3*N + 1)
    for i in range(N):
        ang = 2 * np.pi * i / N
        params[3*i] = 3.0 * np.cos(ang)
        params[3*i+1] = 3.0 * np.sin(ang)
        params[3*i+2] = ang * 180 / np.pi
    params[-1] = 6.0
    initial_guesses.append(params.copy())
    
    # Random starts
    for seed in range(20):
        rng = np.random.RandomState(seed + 1000)
        params = np.zeros(3*N + 1)
        for i in range(N):
            params[3*i] = rng.uniform(-3, 3)
            params[3*i+1] = rng.uniform(-3, 3)
            params[3*i+2] = rng.uniform(0, 60)
        params[-1] = 7.0
        initial_guesses.append(params)
    
    # Optimize from each initial guess
    for idx, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 15.0:
            break
        try_optimize(x0, maxiter=200)
    
    # Polish the best solution
    if best_params is not None and time.time() < deadline - 10.0:
        try_optimize(best_params, maxiter=500)
    
    # Extract final answer
    if best_params is not None:
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner.append([x, y, a])
        center = (0.0, 0.0)
        side = best_L
        angle = 0.0
    else:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 50.0
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
