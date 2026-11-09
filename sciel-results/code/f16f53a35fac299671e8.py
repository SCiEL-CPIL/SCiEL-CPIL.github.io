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

def _min_dist_hexagons(xi, yi, ai, xj, yj, aj):
    """
    Compute minimum distance between two unit hexagons using SAT/edge-edge separation.
    Returns 0.0 if overlapping, else the minimum distance.
    """
    vi = _hex_verts(xi, yi, 1.0, ai)
    vj = _hex_verts(xj, yj, 1.0, aj)
    
    # Quick center-to-center check
    dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    if dc > 2.0:
        return dc - 2.0
    
    # Check if any vertex of vi is inside vj or vice versa
    for v in vi:
        if _point_in_hex(v[0], v[1], xj, yj, 1.0, aj):
            return 0.0
    for v in vj:
        if _point_in_hex(v[0], v[1], xi, yi, 1.0, ai):
            return 0.0
    
    # Compute minimum edge-to-edge distance
    min_d = float('inf')
    for i in range(6):
        ax1, ay1 = vi[i]
        ax2, ay2 = vi[(i+1) % 6]
        for j in range(6):
            bx1, by1 = vj[j]
            bx2, by2 = vj[(j+1) % 6]
            
            # Distance from point ax1 to segment (bx1,by1)-(bx2,by2)
            dx = bx2 - bx1
            dy = by2 - by1
            if dx*dx + dy*dy < 1e-12:
                d = math.sqrt((ax1-bx1)**2 + (ay1-by1)**2)
            else:
                t = ((ax1-bx1)*dx + (ay1-by1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = bx1 + t*dx
                qy = by1 + t*dy
                d = math.sqrt((ax1-qx)**2 + (ay1-qy)**2)
            if d < min_d:
                min_d = d
            
            # Distance from point ax2 to segment (bx1,by1)-(bx2,by2)
            if dx*dx + dy*dy < 1e-12:
                d = math.sqrt((ax2-bx1)**2 + (ay2-by1)**2)
            else:
                t = ((ax2-bx1)*dx + (ay2-by1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = bx1 + t*dx
                qy = by1 + t*dy
                d = math.sqrt((ax2-qx)**2 + (ay2-qy)**2)
            if d < min_d:
                min_d = d
            
            # Distance from point bx1 to segment (ax1,ay1)-(ax2,ay2)
            dx = ax2 - ax1
            dy = ay2 - ay1
            if dx*dx + dy*dy < 1e-12:
                d = math.sqrt((bx1-ax1)**2 + (by1-ay1)**2)
            else:
                t = ((bx1-ax1)*dx + (by1-ay1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = ax1 + t*dx
                qy = ay1 + t*dy
                d = math.sqrt((bx1-qx)**2 + (by1-qy)**2)
            if d < min_d:
                min_d = d
            
            # Distance from point bx2 to segment (ax1,ay1)-(ax2,ay2)
            if dx*dx + dy*dy < 1e-12:
                d = math.sqrt((bx2-ax1)**2 + (by2-ay1)**2)
            else:
                t = ((bx2-ax1)*dx + (by2-ay1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = ax1 + t*dx
                qy = ay1 + t*dy
                d = math.sqrt((bx2-qx)**2 + (by2-qy)**2)
            if d < min_d:
                min_d = d
    
    return min_d

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

    # Check overlap
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            
            dist = _min_dist_hexagons(xi, yi, ai, xj, yj, aj)
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

def _required_side(inner):
    """Compute the minimum outer side length needed to contain the given inner hexagons."""
    max_r = 0.0
    for (xi, yi, ai) in inner:
        ai_rad = math.radians(ai)
        verts = _hex_verts(xi, yi, 1.0, ai_rad)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    return max_r / math.cos(math.radians(30))

def _spectral_packing(inner, t_start, time_limit):
    """
    Use spectral graph optimization to refine the packing.
    Models the contact graph and uses Laplacian eigenvalue spread to find compact arrangements.
    """
    n = len(inner)
    rng = np.random.RandomState(42)
    
    # Build initial contact graph
    def _build_contact_graph(inner, threshold=0.05):
        """Build contact graph where edge exists if hexagons are within threshold distance."""
        graph = np.zeros((n, n))
        for i in range(n):
            xi, yi, ai = inner[i]
            ai_rad = math.radians(ai)
            for j in range(i+1, n):
                xj, yj, aj = inner[j]
                aj_rad = math.radians(aj)
                dist = _min_dist_hexagons(xi, yi, ai_rad, xj, yj, aj_rad)
                if dist < threshold:
                    graph[i, j] = 1.0
                    graph[j, i] = 1.0
        return graph
    
    def _laplacian_eigenvalue_spread(graph, inner):
        """
        Compute a spectral objective: minimize the spread of Laplacian eigenvalues
        while respecting distance constraints.
        """
        # Compute graph Laplacian
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    L[i, i] = graph[i, j].sum()
                else:
                    L[i, j] = -graph[i, j]
        
        # Get eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(L)
        except:
            return 1e6
        
        # Spectral spread: difference between largest and smallest non-zero eigenvalue
        nonzero = eigvals[eigvals > 1e-6]
        if len(nonzero) < 2:
            return 1e6
        
        spread = nonzero[-1] - nonzero[0]
        
        # Add penalty for distance violations
        pen = 0.0
        for i in range(n):
            xi, yi, ai = inner[i]
            ai_rad = math.radians(ai)
            for j in range(i+1, n):
                xj, yj, aj = inner[j]
                aj_rad = math.radians(aj)
                dist = _min_dist_hexagons(xi, yi, ai_rad, xj, yj, aj_rad)
                if dist < 0.0:
                    pen += abs(dist) ** 2 * 100.0
        
        # Boundary penalty
        S = _required_side(inner)
        r_out = _hex_inradius(S)
        c30 = math.cos(math.radians(30))
        for i in range(n):
            xi, yi, ai = inner[i]
            ai_rad = math.radians(ai)
            verts = _hex_verts(xi, yi, 1.0, ai_rad)
            for v in verts:
                d1 = abs(c30 * v[0] + 0.5 * v[1])
                d2 = abs(v[1])
                d3 = abs(-c30 * v[0] + 0.5 * v[1])
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    pen += (d_max - r_out) ** 2
        
        return spread + pen
    
    # Spectral refinement: perturb positions to reduce spectral spread
    best_inner = list(inner)
    best_obj = _laplacian_eigenvalue_spread(_build_contact_graph(best_inner), best_inner)
    
    for iteration in range(20):
        if time.time() > time_limit:
            break
        
        # Perturb positions slightly
        perturbed = []
        for (xi, yi, ai) in best_inner:
            nx = xi + rng.randn() * 0.1
            ny = yi + rng.randn() * 0.1
            na = ai + rng.randn() * 5.0
            perturbed.append((nx, ny, na))
        
        obj = _laplacian_eigenvalue_spread(_build_contact_graph(perturbed), perturbed)
        
        if obj < best_obj:
            best_obj = obj
            best_inner = perturbed
    
    return best_inner

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Generate initial valid configuration
    init_pos = _honeycomb_positions()
    
    # Calculate required outer side for initial config
    s_init = _required_side(init_pos)
    s_init *= 1.01
    
    best_result = None
    best_S = s_init
    
    # First pass: spectral packing refinement
    if _remaining() > 30.0:
        spectral_inner = _spectral_packing(init_pos, t_start, _DEADLINE - 15.0)
        S_spec = _required_side(spectral_inner)
        if _is_valid_config(spectral_inner, (0.0, 0.0), S_spec, 0.0):
            if S_spec < best_S:
                best_S = S_spec
                best_result = spectral_inner
    
    # Second pass: local optimization with smooth distance metric
    if _remaining() > 20.0:
        # Prepare initial parameters for optimization
        start_inner = best_result if best_result else init_pos
        p0 = np.zeros(3*n + 1)
        for i in range(n):
            p0[3*i] = start_inner[i][0]
            p0[3*i+1] = start_inner[i][1]
            p0[3*i+2] = start_inner[i][2]
        p0[3*n] = best_S
        
        def objective(p):
            x = p[0:3*n:3]
            y = p[1:3*n:3]
            angles = p[2:3*n:3]
            S = p[3*n]
            
            if S < 1.0:
                return 1e6
                
            pen = 0.0
            r_out = _hex_inradius(S)
            c30 = math.cos(math.radians(30))
            
            # Boundary penalty
            for i in range(n):
                verts = _hex_verts(x[i], y[i], 1.0, angles[i])
                for v in verts:
                    d1 = abs(c30 * v[0] + 0.5 * v[1])
                    d2 = abs(v[1])
                    d3 = abs(-c30 * v[0] + 0.5 * v[1])
                    d_max = max(d1, d2, d3)
                    if d_max > r_out:
                        pen += (d_max - r_out)**2
                        
            # Overlap penalty using smooth distance metric
            for i in range(n):
                for j in range(i+1, n):
                    dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dc > 2.0:
                        continue
                    
                    dist = _min_dist_hexagons(x[i], y[i], angles[i], x[j], y[j], angles[j])
                    if dist < 0.01:
                        pen += max(0, 0.01 - dist) ** 2 * 1000.0
            
            return 0.5 * S * S + pen
        
        best_val = float('inf')
        best_p = p0.copy()
        
        for trial in range(5):
            if _remaining() < 15.0:
                break
                
            p_init = p0.copy()
            if trial > 0:
                noise = 0.05 + 0.02 * trial
                p_init[0:3*n] += np.random.RandomState(42 + trial).randn(3*n) * noise
                p_init[3*n] = best_S * (1.0 + np.random.RandomState(42 + trial).rand()*0.05)
                
            try:
                res = minimize(objective, p_init, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
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
            if S < best_S:
                best_result = inner
                best_S = S
    
    # Final fallback to initial valid configuration
    if best_result is None:
        inner_init = [[float(p[0]), float(p[1]), float(p[2])] for p in init_pos]
        return inner_init, (0.0, 0.0), float(s_init), 0.0
    
    return best_result, (0.0, 0.0), float(best_S), 0.0
