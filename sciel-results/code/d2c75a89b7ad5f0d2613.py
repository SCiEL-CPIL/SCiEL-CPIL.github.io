"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _hex_verts(cx, cy, s, rot):
    """Get the 6 vertices of a hexagon centered at (cx,cy) with side s and rotation rot (radians)."""
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot (radians)."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _support(A, d):
    """Find vertex of polygon A (array of shape (6,2)) that maximizes dot product with direction d."""
    dots = A @ d
    return A[np.argmax(dots)]

def _gjk_distance(A, B, max_iter=50):
    """
    Compute minimum distance between two convex polygons A and B (each shape (6,2)) using GJK.
    Returns the distance (0 if overlapping).
    """
    d = np.array([1.0, 0.0])
    simplex = []
    
    for _ in range(max_iter):
        a = _support(A, d)
        b = _support(B, -d)
        p = a - b
        
        if np.dot(p, d) < 0:
            break
        
        simplex.append(p)
        
        if len(simplex) == 1:
            d = -simplex[0]
            if np.dot(d, d) < 1e-12:
                return 0.0
        elif len(simplex) == 2:
            p0 = simplex[0]
            p1 = simplex[1]
            v = p1 - p0
            w = -p0
            t = np.dot(w, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
            if t >= 0 and t <= 1:
                closest = p0 + t * v
                if np.dot(closest, closest) < 1e-12:
                    return 0.0
                d = np.array([-v[1], v[0]])
                if np.dot(d, closest) > 0:
                    d = -d
            else:
                if t < 0:
                    d = -p0
                    simplex = [p0]
                else:
                    d = -p1
                    simplex = [p1]
        else:
            p0 = simplex[0]
            p1 = simplex[1]
            p2 = simplex[2]
            
            v0 = p2 - p0
            v1 = p1 - p0
            w = -p0
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(w, v0)
            d21 = np.dot(w, v1)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                d = -p0
                simplex = [p0]
                continue
            
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            
            if u >= 0 and v >= 0 and u + v <= 1:
                return 0.0
            elif u >= 0 and v < 0:
                d = -p0
                simplex = [p0]
            elif u < 0 and v >= 0:
                d = -p0
                simplex = [p0]
            elif u >= 0 and v >= 0 and u + v > 1:
                d = -(p1 + p2) / 2
                simplex = [p1, p2]
            elif u < 0 and v < 0:
                d = -p0
                simplex = [p0]
            else:
                d = -p0
                simplex = [p0]
    
    if len(simplex) == 1:
        return np.linalg.norm(simplex[0])
    elif len(simplex) == 2:
        p0 = simplex[0]
        p1 = simplex[1]
        v = p1 - p0
        t = np.dot(-p0, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
        t = max(0, min(1, t))
        closest = p0 + t * v
        return np.linalg.norm(closest)
    else:
        return min(np.linalg.norm(p) for p in simplex)

def _min_dist_hexagons(xi, yi, ai, xj, yj, aj):
    """Compute minimum distance between two unit hexagons with given positions and angles."""
    vi = _hex_verts(xi, yi, 1.0, ai)
    vj = _hex_verts(xj, yj, 1.0, aj)
    return _gjk_distance(vi, vj)

def _is_valid_config(inner, center, side, angle):
    """
    Strictly validate the configuration.
    inner: list of (x, y, angle_deg)
    center: (cx, cy)
    side: outer side length
    angle: outer rotation (radians)
    """
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # Check boundary
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # Check overlap using GJK
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Quick reject
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            # Use GJK to check for overlap
            dist = _gjk_distance(vi, vj)
            if dist < 1e-9:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    # Ring 2: 4 hexagons at distance 4, alternating angles
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def _dynamics_simulate(init_pos, S, dt=0.005, n_steps=5000, damping=0.95, 
                       attract_strength=0.5, contact_stiffness=50.0,
                       time_limit=None, t_start=None):
    """
    Rigid-body dynamics simulation using velocity-Verlet integration.
    Inner hexagons are attracted toward center and repel each other on contact.
    Returns final positions and angles.
    """
    if time_limit is None or t_start is None:
        t_start = time.time()
        time_limit = _DEADLINE - 5.0
    
    n = len(init_pos)
    # State: [x1, y1, a1, x2, y2, a2, ..., vx1, vy1, va1, vx2, vy2, va2, ...]
    state = np.zeros(6 * n)
    for i in range(n):
        state[3*i] = init_pos[i][0]
        state[3*i+1] = init_pos[i][1]
        state[3*i+2] = math.radians(init_pos[i][2])
    
    # Velocity
    vel = np.zeros(6 * n)
    
    # Precompute some constants
    c30 = math.cos(math.radians(30))
    r_out = _hex_inradius(S)
    
    def _get_pos_angles(state):
        x = state[0:3*n:3]
        y = state[1:3*n:3]
        a = state[2:3*n:3]
        return x, y, a
    
    def _compute_forces(x, y, a):
        """Compute forces and torques on each hexagon."""
        F = np.zeros(n)
        Fy = np.zeros(n)
        T = np.zeros(n)
        
        # Attraction to center
        for i in range(n):
            r = math.sqrt(x[i]**2 + y[i]**2)
            if r > 1e-6:
                F[i] -= attract_strength * x[i] / r
                Fy[i] -= attract_strength * y[i] / r
        
        # Contact repulsion between hexagons
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.5:
                    continue
                dist = _min_dist_hexagons(x[i], y[i], a[i], x[j], y[j], a[j])
                if dist < 0.05:
                    # Repulsive force
                    if dc > 1e-6:
                        fx = (x[i] - x[j]) / dc
                        fy = (y[i] - y[j]) / dc
                        force = contact_stiffness * max(0, 0.05 - dist)
                        F[i] += force * fx
                        Fy[i] += force * fy
                        F[j] -= force * fx
                        Fy[j] -= force * fy
                    # Small torque to help rotate away
                    T[i] += 0.1 * (a[i] - a[j])
                    T[j] -= 0.1 * (a[i] - a[j])
        
        # Boundary constraint: keep vertices inside outer hexagon
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, a[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(c30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-c30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    # Push back toward center
                    r = math.sqrt(x[i]**2 + y[i]**2)
                    if r > 1e-6:
                        penalty = contact_stiffness * (d_max - r_out)
                        F[i] += penalty * x[i] / r
                        Fy[i] += penalty * y[i] / r
        
        return F, Fy, T
    
    # Velocity-Verlet integration
    F, Fy, T = _compute_forces(*_get_pos_angles(state))
    
    for step in range(n_steps):
        if time.time() > time_limit:
            break
        
        # Update positions
        for i in range(n):
            state[3*i] += vel[3*i] * dt + 0.5 * F[i] * dt * dt
            state[3*i+1] += vel[3*i+1] * dt + 0.5 * Fy[i] * dt * dt
            state[3*i+2] += vel[3*i+2] * dt + 0.5 * T[i] * dt * dt
        
        # Compute new forces
        x, y, a = _get_pos_angles(state)
        F_new, Fy_new, T_new = _compute_forces(x, y, a)
        
        # Update velocities
        for i in range(n):
            vel[3*i] = (vel[3*i] + 0.5 * (F[i] + F_new[i]) * dt) * damping
            vel[3*i+1] = (vel[3*i+1] + 0.5 * (Fy[i] + Fy_new[i]) * dt) * damping
            vel[3*i+2] = (vel[3*i+2] + 0.5 * (T[i] + T_new[i]) * dt) * damping
        
        F, Fy, T = F_new, Fy_new, T_new
    
    x, y, a = _get_pos_angles(state)
    return list(zip(x.tolist(), y.tolist(), [math.degrees(ai) for ai in a]))

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Generate initial valid configuration
    init_pos = _honeycomb_positions()
    
    # Calculate required outer side for initial config
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / math.cos(math.radians(30))
    s_init *= 1.01
    
    best_result = None
    best_S = s_init
    
    # Try dynamics simulation with different parameters
    rng = np.random.RandomState(42)
    
    for trial in range(6):
        if _remaining() < 30.0:
            break
        
        # Perturb initial positions
        perturbed = []
        for (xi, yi, ai) in init_pos:
            noise = 0.1 + 0.05 * trial
            nx = xi + rng.randn() * noise
            ny = yi + rng.randn() * noise
            na = ai + rng.randn() * 10
            perturbed.append((nx, ny, na))
        
        # Try different outer sizes
        for S_frac in [1.0, 0.98, 0.96]:
            S_try = s_init * S_frac
            if S_try < 2.5:
                continue
            
            if _remaining() < 15.0:
                break
            
            # Run dynamics simulation
            try:
                result = _dynamics_simulate(perturbed, S_try, 
                                           dt=0.005, n_steps=3000,
                                           damping=0.95, attract_strength=0.3,
                                           contact_stiffness=30.0,
                                           time_limit=_DEADLINE - 5.0,
                                           t_start=t_start)
                
                # Check if result is valid
                if _is_valid_config(result, (0.0, 0.0), S_try, 0.0):
                    if S_try < best_S:
                        best_S = S_try
                        best_result = result
            except Exception:
                pass
    
    # If dynamics didn't find anything better, use optimization as fallback
    if best_result is None:
        # Prepare initial parameters for optimization
        p0 = np.zeros(3*n + 1)
        for i in range(n):
            p0[3*i] = init_pos[i][0]
            p0[3*i+1] = init_pos[i][1]
            p0[3*i+2] = 0.0
        p0[3*n] = s_init
        
        def objective(p):
            x = p[0:3*n:3]
            y = p[1:3*n:3]
            angles = p[2:3*n:3]
            S = p[3*n]
            
            if S < 1.0:
                return 1e6
                
            pen = 0.0
            r_out = _hex_inradius(S)
            
            # Boundary penalty
            for i in range(n):
                verts = _hex_verts(x[i], y[i], 1.0, angles[i])
                for v in verts:
                    dx = v[0]
                    dy = v[1]
                    d1 = abs(c30 * dx + 0.5 * dy)
                    d2 = abs(dy)
                    d3 = abs(-c30 * dx + 0.5 * dy)
                    d_max = max(d1, d2, d3)
                    if d_max > r_out:
                        pen += (d_max - r_out)**2
                        
            # Overlap penalty using GJK
            for i in range(n):
                for j in range(i+1, n):
                    dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dc > 2.0:
                        continue
                    
                    vi = _hex_verts(x[i], y[i], 1.0, angles[i])
                    vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                    
                    dist = _gjk_distance(vi, vj)
                    if dist < 0.01:
                        pen += max(0, 0.01 - dist) ** 2 * 1000.0
            
            return 0.5 * S * S + pen
        
        best_val = float('inf')
        best_p = p0.copy()
        
        for trial in range(5):
            if _remaining() < 20.0:
                break
                
            p_init = p0.copy()
            if trial > 0:
                noise = 0.05 + 0.02 * trial
                p_init[0:3*n] += rng.randn(3*n) * noise
                p_init[3*n] = s_init * (1.0 + rng.rand()*0.05)
                
            try:
                res = minimize(objective, p_init, method='Nelder-Mead',
                              options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_val:
                    best_val = res.fun
                    best_p = res.x.copy()
            except Exception:
                pass
        
        x = best_p[0:3*n:3]
        y = best_p[1:3*n:3]
        angles = best_p[2:3*n:3]
        S = best_p[3*n]
        
        inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
        
        if _is_valid_config(inner, (0.0, 0.0), S, 0.0):
            best_result = inner
            best_S = S
    
    # Final fallback to initial valid configuration
    if best_result is None:
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, (0.0, 0.0), float(s_init), 0.0
    
    return best_result, (0.0, 0.0), float(best_S), 0.0
