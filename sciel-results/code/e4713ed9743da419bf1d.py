"""Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 120.0
START_TIME = time.time()

# Precompute unit hexagon vertices (pointy-top, side=1)
# Vertices at angles 30, 90, 150, 210, 270, 330 degrees
_HEX_ANGLES = np.array([np.pi/6 + k*np.pi/3 for k in range(6)])
_HEX_VERTS = np.column_stack([np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)])

# Outer hexagon vertices for side length s, center (cx,cy), rotation angle theta
# Vertices at angles theta + k*60 degrees, distance s from center
def outer_vertices(cx, cy, s, theta):
    angles = theta + np.array([k*np.pi/3 for k in range(6)])
    return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])

def inner_hex_verts(x, y, angle_deg):
    """Get vertices of a unit hexagon at (x,y) with rotation angle_deg."""
    a = np.radians(angle_deg)
    c, s_ = np.cos(a), np.sin(a)
    R = np.array([[c, -s_], [s_, c]])
    return _HEX_VERTS @ R.T + np.array([x, y])

def point_in_hex(px, py, cx, cy, s, theta):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side s, rotation theta."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    a = -theta
    c, s_ = np.cos(a), np.sin(a)
    lx = c*dx - s_*dy
    ly = s_*dx + c*dy
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 at distance s
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with circumradius s, the inradius is s*cos(30) = s*sqrt(3)/2
    # Actually for side length s, circumradius = s, inradius = s*sqrt(3)/2
    # The 6 sides are at distance inradius from center, at angles 30, 90, 150, 210, 270, 330
    inr = s * math.sqrt(3) / 2
    for k in range(6):
        ang = np.pi/6 + k*np.pi/3
        nx, ny = np.cos(ang), np.sin(ang)
        if lx*nx + ly*ny > inr:
            return False
    return True

def hexagon_containment_penalty(x, y, angle_deg, cx, cy, s, theta):
    """Penalty for inner hexagon not fully inside outer hexagon.
    Returns 0 if fully inside, else sum of squared penetration depths."""
    verts = inner_hex_verts(x, y, angle_deg)
    inr = s * math.sqrt(3) / 2
    penalty = 0.0
    for vx, vy in verts:
        dx, dy = vx - cx, vy - cy
        a = -theta
        c, s_ = np.cos(a), np.sin(a)
        lx = c*dx - s_*dy
        ly = s_*dx + c*dy
        for k in range(6):
            ang = np.pi/6 + k*np.pi/3
            nx, ny = np.cos(ang), np.sin(ang)
            dist = lx*nx + ly*ny
            if dist > inr:
                penalty += (dist - inr)**2
    return penalty

def hexagon_overlap_penalty(x1, y1, a1, x2, y2, a2):
    """Penalty for overlap between two unit hexagons.
    Uses distance between centers as a proxy; for non-overlapping unit hexagons
    the minimum center distance depends on relative orientation.
    For identical orientation, min distance is 2*sqrt(3)/2 = sqrt(3) ~ 1.732 (side by side)
    or 2*1 = 2 (vertex to vertex). Actually for two unit hexagons side by side,
    center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3).
    
    A simpler approach: check if any vertex of one is inside the other, or if edges intersect.
    For efficiency, use a soft penalty based on center distance with a threshold.
    The minimum non-overlapping center distance for two unit hexagons is sqrt(3) (when sharing an edge).
    """
    dx, dy = x1-x2, y1-y2
    dist = math.sqrt(dx*dx + dy*dy)
    # Minimum center distance for non-overlapping unit hexagons is sqrt(3)
    min_dist = math.sqrt(3)
    if dist < min_dist:
        return (min_dist - dist)**2
    return 0.0

def total_penalty(params):
    """
    params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, cx,cy,s, theta]
    Total 3*N + 4 = 40 parameters.
    """
    n_params = 3*N + 4
    if len(params) != n_params:
        return 1e10
    
    inner = params[:3*N].reshape(N, 3)
    cx, cy, s, theta = params[3*N:]
    
    if s <= 0:
        return 1e10
    
    penalty = 0.0
    
    # Containment: each inner hex must be inside outer
    for i in range(N):
        x, y, a = inner[i]
        penalty += hexagon_containment_penalty(x, y, a, cx, cy, s, theta)
    
    # Non-overlap: each pair of inner hexes must not overlap
    for i in range(N):
        for j in range(i+1, N):
            penalty += hexagon_overlap_penalty(
                inner[i][0], inner[i][1], inner[i][2],
                inner[j][0], inner[j][1], inner[j][2]
            )
    
    return penalty

def objective(params):
    """Objective: outer side length + large penalty for constraint violations."""
    n_params = 3*N + 4
    if len(params) != n_params:
        return 1e10
    
    s = params[3*N+2]
    if s <= 0:
        return 1e10
    
    pen = total_penalty(params)
    if pen > 1e-6:
        return 1e6 + s + 1000*pen
    return s

def make_initial_guess_hex_lattice():
    """Create initial guess with hexagonal lattice arrangement."""
    # Arrange 12 hexagons in a hexagonal pattern
    # Hex lattice: positions for a hexagonal cluster
    # For 12 hexagons, use a 2-ring hexagonal pattern (1 center + 6 ring + 5 more)
    # Or use a rectangular grid of hexagons
    
    # Simple approach: 3x4 grid with hexagonal offsets
    positions = []
    dx = math.sqrt(3) * 1.05  # slight spacing
    dy = 1.5 * 1.05
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx/2
            y = row * dy
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    # Outer hexagon: large enough to contain all
    max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in positions)
    s0 = max_r + 2.0
    
    params = [v for p in positions for v in p]
    params += [0.0, 0.0, s0, 0.0]
    return np.array(params)

def make_initial_guess_circle():
    """Initial guess with hexagons arranged in a circle."""
    positions = []
    r = 2.5
    for i in range(N):
        a = 2*math.pi*i/N
        positions.append([r*math.cos(a), r*math.sin(a), 0.0])
    
    s0 = r + 2.5
    params = [v for p in positions for v in p]
    params += [0.0, 0.0, s0, 0.0]
    return np.array(params)

def make_initial_guess_line():
    """Initial guess with hexagons in a line."""
    positions = []
    dx = math.sqrt(3) * 1.1
    for i in range(N):
        x = (i - N/2 + 0.5) * dx
        positions.append([x, 0.0, 0.0])
    
    max_r = max(abs(p[0]) for p in positions)
    s0 = max_r + 2.0
    params = [v for p in positions for v in p]
    params += [0.0, 0.0, s0, 0.0]
    return np.array(params)

def bounds():
    """Bounds for differential evolution."""
    b = []
    for _ in range(N):
        b.append([-15, 15])  # x
        b.append([-15, 15])  # y
        b.append([0, 180])   # angle
    b.append([-5, 5])  # cx
    b.append([-5, 5])  # cy
    b.append([1.0, 20.0])  # s
    b.append([0, 60])  # theta
    return b

def construct():
    global START_TIME, DEADLINE
    best_params = None
    best_score = float('inf')
    
    deadline = START_TIME + DEADLINE - 5.0  # 5s safety margin
    
    # Try multiple initializations
    initializers = [
        make_initial_guess_hex_lattice,
        make_initial_guess_circle,
        make_initial_guess_line,
    ]
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(3):
        def make_random():
            positions = []
            for i in range(N):
                x = rng.uniform(-4, 4)
                y = rng.uniform(-4, 4)
                a = rng.uniform(0, 180)
                positions.append([x, y, a])
            s0 = 8.0
            params = [v for p in positions for v in p]
            params += [0.0, 0.0, s0, 0.0]
            return np.array(params)
        initializers.append(make_random)
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        x0 = init_func()
        
        # First try L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          bounds=bounds(), 
                          options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
        
        # Then try Nelder-Mead from the L-BFGS result
        if best_params is not None:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
        
        if time.time() > deadline:
            break
    
    # If we still have time, try differential evolution for global search
    if time.time() < deadline - 20.0:
        try:
            b = bounds()
            de_res = differential_evolution(objective, b, seed=42, maxiter=50, 
                                           popsize=15, tol=1e-8, polish=True)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish if we have time
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            polish_res = minimize(objective, best_params, method='L-BFGS-B',
                                bounds=bounds(),
                                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12})
            if polish_res.fun < best_score:
                best_score = polish_res.fun
                best_params = polish_res.x.copy()
        except Exception:
            pass
    
    # Extract result
    if best_params is None:
        # Fallback: return a valid (if not optimal) configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a = float(best_params[3*i+2])
        inner.append([x, y, a])
    
    cx = float(best_params[3*N])
    cy = float(best_params[3*N+1])
    s = float(best_params[3*N+2])
    theta = float(best_params[3*N+3])
    
    return inner, (cx, cy), s, theta

# Set start time when module is loaded
START_TIME = time.time()
