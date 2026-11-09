"""Optimized hexagon packing: 11 unit hexagons in smallest regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Get 6 vertices of a regular hexagon with center (cx,cy), rotation, side s."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.pi / 3 * i
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, angle_deg, s=1.0):
    """Check if point (px,py) is inside hexagon with center (cx,cy), rotation, side s."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    a = np.radians(-angle_deg)
    lx = dx * np.cos(a) - dy * np.sin(a)
    ly = dx * np.sin(a) + dy * np.cos(a)
    # For a regular hexagon with side s, in local frame with flat top:
    # The hexagon extends to s in x, and s*sqrt(3)/2 in y (for pointy-top)
    # Actually for our vertex definition, the hexagon has "radius" = s (center to vertex)
    # The inradius is s * cos(pi/6) = s * sqrt(3)/2
    # In local frame, vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # So the hexagon is defined by |lx| <= s and |ly| <= s*sqrt(3)/2 approximately
    # More precisely, check against each edge
    verts = _hex_vertices(0, 0, 0, s)
    # Point is inside if it's on the same side of all edges
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Edge vector
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        # Normal (inward)
        nx, ny = -ey, ex
        # Check if point is on inward side
        # The inward normal should point toward center
        # Center is at origin, v1 is a vertex
        # Inward direction from edge midpoint toward center
        mx, my = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2
        # Normal pointing inward
        nx_in, ny_in = -mx, -my
        # Make sure normal points inward
        dot = nx * nx_in + ny * ny_in
        if dot < 0:
            nx, ny = -nx, -ny
        # Check point relative to edge
        val = (lx - v1[0]) * nx + (ly - v1[1]) * ny
        if val < -1e-10:
            return False
    return True

def _hex_overlap_penalty(hex1_params, hex2_params):
    """Compute overlap penalty between two unit hexagons."""
    cx1, cy1, a1 = hex1_params[0], hex1_params[1], hex1_params[2]
    cx2, cy2, a2 = hex2_params[0], hex2_params[1], hex2_params[2]
    
    # Quick rejection: if centers are far apart, no overlap
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = np.sqrt(dx*dx + dy*dy)
    if dist > 2.0:  # Max distance for overlap of unit hexagons
        return 0.0
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    verts1 = _hex_vertices(cx1, cy1, a1)
    verts2 = _hex_vertices(cx2, cy2, a2)
    
    overlap = 0.0
    # Check vertices of hex1 in hex2
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, a2):
            overlap += 1.0
    # Check vertices of hex2 in hex1
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, a1):
            overlap += 1.0
    
    return overlap

def _outer_hex_contains_point(px, py, ocx, ocy, oside, oangle):
    """Check if point is inside outer hexagon."""
    return _point_in_hex(px, py, ocx, ocy, oangle, oside)

def _objective(params):
    """Objective: outer side length + penalties for violations."""
    if time.time() > DEADLINE:
        return 1e10
    
    # params: [ox, oy, oside, oangle, x1, y1, a1, x2, y2, a2, ...]
    # 4 outer params + 3*N inner params = 4 + 33 = 37
    ox, oy, oside, oangle = params[0], params[1], params[2], params[3]
    
    # Ensure oside is positive
    if oside < 0.5:
        oside = 0.5
    
    penalty = 0.0
    
    # Check all inner hexagons are inside outer hexagon
    for i in range(N):
        ix = params[4 + 3*i]
        iy = params[4 + 3*i + 1]
        ia = params[4 + 3*i + 2]
        
        # Check center is inside
        if not _outer_hex_contains_point(ix, iy, ox, oy, oside, oangle):
            # Compute distance to boundary (approximate)
            dx, dy = ix - ox, iy - oy
            dist_center = np.sqrt(dx*dx + dy*dy)
            # Inradius of outer hex
            inradius = oside * np.sqrt(3) / 2
            if dist_center > inradius:
                penalty += (dist_center - inradius) ** 2 * 10
    
    # Check all inner hexagon vertices are inside outer hexagon
    for i in range(N):
        ix = params[4 + 3*i]
        iy = params[4 + 3*i + 1]
        ia = params[4 + 3*i + 2]
        
        verts = _hex_vertices(ix, iy, ia)
        for v in verts:
            if not _outer_hex_contains_point(v[0], v[1], ox, oy, oside, oangle):
                # Approximate penalty
                dx, dy = v[0] - ox, v[1] - oy
                dist = np.sqrt(dx*dx + dy*dy)
                inradius = oside * np.sqrt(3) / 2
                if dist > inradius:
                    penalty += (dist - inradius) ** 2 * 5
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            h1 = (params[4+3*i], params[4+3*i+1], params[4+3*i+2])
            h2 = (params[4+3*j], params[4+3*j+1], params[4+3*j+2])
            penalty += _hex_overlap_penalty(h1, h2) * 10
    
    return oside + penalty

def _initial_guess():
    """Create initial guess: hexagonal arrangement of 11 hexagons."""
    # Center at origin, outer hexagon centered at origin
    # Arrange 11 hexagons in a pattern: center + 6 around + 4 more
    # For unit hexagons, adjacent centers are distance sqrt(3) apart (for edge-to-edge)
    # Actually for regular hexagons with side 1, the distance between centers of adjacent
    # hexagons sharing an edge is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    
    d = np.sqrt(3)  # distance between adjacent hexagon centers
    
    # Hexagonal grid positions (axial coordinates)
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance d
    # We need 4 more from ring 2
    
    positions = []
    # Center
    positions.append((0, 0, 0))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append((x, y, 0))
    
    # Ring 2: pick 4 from the 12 positions
    # Ring 2 positions are at distance 2*d or d*sqrt(3)
    # Axial coordinates for ring 2: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (0,-3), (-1,-2)
    # Let's use a subset that fits nicely
    ring2_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial to Cartesian for hex grid
            x = d * (q + r/2)
            y = d * np.sqrt(3)/2 * r
            # Only add if not already in positions and at proper distance
            if not any(np.sqrt((x-px)**2 + (y-py)**2) < 0.1 for px, py, _ in positions):
                ring2_candidates.append((x, y, 0))
    
    # Pick 4 from ring 2 that are closest to center
    ring2_candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    for pos in ring2_candidates[:4]:
        positions.append(pos)
    
    # Ensure we have exactly N positions
    while len(positions) < N:
        positions.append((0, 0, 0))  # fallback
    
    positions = positions[:N]
    
    # Build params: [ox, oy, oside, oangle, x1, y1, a1, ...]
    # Estimate outer side length: the extent of our arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    max_extent = max(max(xs) - min(xs), max(ys) - min(ys))
    oside_guess = max_extent / np.sqrt(3) + 1.5  # rough estimate
    
    params = [0.0, 0.0, oside_guess, 0.0]
    for p in positions:
        params.extend([p[0], p[1], p[2]])
    
    return params

def construct():
    """Construct the packing: returns (inner, center, side, angle)."""
    start_time = time.time()
    
    # Try multiple random starts with local optimization
    best_params = None
    best_score = float('inf')
    
    # Initial structured guess
    init_params = _initial_guess()
    
    # Run optimization from initial guess
    result = minimize(_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result.fun < best_score:
        best_score = result.fun
        best_params = result.x.copy()
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    num_restarts = min(5, int(remaining / 15))  # ~15s per restart
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        # Perturb the initial guess
        perturbation = np.random.randn(4 + 3*N) * 0.5
        perturbation[2] = abs(perturbation[2]) * 0.3 + 2.0  # keep oside reasonable
        perturbation[3] = perturbation[3] * 0.1  # keep angle small
        
        params = init_params + perturbation
        params[2] = max(params[2], 2.0)  # minimum outer side
        
        result = minimize(_objective, params, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    if best_params is None:
        best_params = init_params
    
    # Extract results
    ox, oy, oside, oangle = best_params[0], best_params[1], best_params[2], best_params[3]
    
    inner = []
    for i in range(N):
        ix = best_params[4 + 3*i]
        iy = best_params[4 + 3*i + 1]
        ia = best_params[4 + 3*i + 2]
        # Normalize angle to [0, 360)
        ia = ia % 360
        inner.append([float(ix), float(iy), float(ia)])
    
    # Normalize outer angle
    oangle = float(oangle % 360)
    oside = float(oside)
    
    return inner, (float(ox), float(oy)), oside, oangle
