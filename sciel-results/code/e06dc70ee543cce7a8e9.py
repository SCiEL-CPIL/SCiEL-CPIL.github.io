"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    def get_vertices(cx, cy, angle_deg):
        """Returns list of 6 (x, y) vertices for a unit regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, angle_deg):
        """Check if point (px, py) is strictly inside the hexagon."""
        # Transform to local coords
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Inradius is sqrt(3)/2
        inradius = sqrt3 / 2
        # Check 6 half-planes
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inradius:
                return False
        return True

    def dist_point_to_segment(px, py, ax, ay, bx, by):
        """Minimum distance from point (px, py) to segment (ax, ay)-(bx, by)."""
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        
        c2 = vx * vx + vy * vy
        if c1 >= c2:
            return math.hypot(px - bx, py - by)
        
        b = c1 / c2
        px_proj = ax + b * vx
        py_proj = ay + b * vy
        return math.hypot(px - px_proj, py - py_proj)

    def dist_point_to_hexagon(px, py, cx, cy, angle_deg):
        """Minimum distance from point (px, py) to the boundary of the hexagon.
        If point is inside, returns 0 (or negative if we wanted depth, but we use 0 for simplicity in penalty logic? 
        No, for penalty we want to know HOW FAR inside it is. 
        Let's return the minimum distance to the boundary. If inside, this is the distance to the nearest edge."""
        verts = get_vertices(cx, cy, angle_deg)
        min_dist = float('inf')
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            d = dist_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
        return min_dist

    def overlap_penalty(c1, a1, c2, a2):
        """Calculate a continuous penalty for overlap between two hexagons."""
        # Quick rejection: max distance between centers is 2.0
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        if dx*dx + dy*dy > 4.0 + 1e-9:
            return 0.0
            
        penalty = 0.0
        verts1 = get_vertices(c1[0], c1[1], a1)
        verts2 = get_vertices(c2[0], c2[1], a2)
        
        # Check vertices of 1 in 2
        for p in verts1:
            if point_in_hexagon(p[0], p[1], c2[0], c2[1], a2):
                # Distance to boundary of 2
                d = dist_point_to_hexagon(p[0], p[1], c2[0], c2[1], a2)
                penalty += d * d
                
        # Check vertices of 2 in 1
        for p in verts2:
            if point_in_hexagon(p[0], p[1], c1[0], c1[1], a1):
                # Distance to boundary of 1
                d = dist_point_to_hexagon(p[0], p[1], c1[0], c1[1], a1)
                penalty += d * d
                
        return penalty

    def boundary_penalty(c, a, S):
        """Penalty if hexagon at c, a is outside outer hexagon of side S, center (0,0)."""
        verts = get_vertices(c[0], c[1], a)
        inradius_outer = S * sqrt3 / 2
        penalty = 0.0
        for p in verts:
            # Distance to boundary of outer hexagon
            # If outside, distance is positive. We want to penalize this.
            # Calculate max projection violation
            max_viol = 0.0
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = p[0] * nx + p[1] * ny
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    if viol > max_viol:
                        max_viol = viol
            penalty += max_viol * max_viol
        return penalty

    def objective(params):
        # params: 2*N positions + N angles + 1 side
        pos = []
        for i in range(N):
            pos.append((params[2*i], params[2*i+1]))
        angs = [params[2*N + i] for i in range(N)]
        S = params[3*N]
        
        if S < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                penalty += overlap_penalty(pos[i], angs[i], pos[j], angs[j])
        
        # Boundary
        for i in range(N):
            penalty += boundary_penalty(pos[i], angs[i], S)
            
        # We want to minimize S, but S is in the objective.
        # To ensure feasibility, we weight the penalty heavily.
        # However, DE minimizes the sum. If penalty is large, it pushes S up or moves hexagons.
        # A better formulation: Minimize S + Lambda * Penalty.
        # If Penalty > 0, it dominates.
        return S + 100.0 * penalty

    # Multi-start DE
    best_solution = None
    best_score = float('inf')
    
    # Define bounds
    # Positions: roughly within [-4, 4]
    # Angles: [0, 360) or (-180, 180]
    # Side: [3.0, 6.0]
    bounds = []
    for i in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((3.0, 6.0))
    
    # Generate initial guesses
    # 1. Grid-like
    # 2. Random
    inits = []
    
    # Grid initialization
    # Hexagonal packing: 3-4-4 or 2-4-4-1?
    # Let's try a few structured inits
    # Init 1: 3-4-4 grid
    rows = [3, 4, 4]
    p_init = []
    a_init = []
    y_off = 0.0
    for count in rows:
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_off
            p_init.append((x, y))
            a_init.append(0.0)
        y_off += 1.5
    # Center it
    xs = [p[0] for p in p_init]
    ys = [p[1] for p in p_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    p_init = [(p[0]-cx, p[1]-cy) for p in p_init]
    inits.append((p_init, a_init))
    
    # Init 2: 4-3-4 grid
    rows = [4, 3, 4]
    p_init = []
    a_init = []
    y_off = 0.0
    for count in rows:
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_off
            p_init.append((x, y))
            a_init.append(0.0)
        y_off += 1.5
    xs = [p[0] for p in p_init]
    ys = [p[1] for p in p_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    p_init = [(p[0]-cx, p[1]-cy) for p in p_init]
    inits.append((p_init, a_init))
    
    # Init 3: Random
    import random
    random.seed(42)
    p_init = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(N)]
    a_init = [random.uniform(0, 360) for _ in range(N)]
    inits.append((p_init, a_init))
    
    # Run DE for each init
    for idx, (p_init, a_init) in enumerate(inits):
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
        for i in range(N):
            x0.append(a_init[i])
        # Estimate S
        max_r = max(math.hypot(p[0], p[1]) for p in p_init)
        x0.append(max_r + 1.5)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=150,
                popsize=30,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=100 + idx,
                polish=True # Use L-BFGS-B for final polish
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception as e:
            pass

    if best_solution is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    pos = []
    for i in range(N):
        pos.append((best_solution[2*i], best_solution[2*i+1]))
    angs = [best_solution[2*N + i] for i in range(N)]
    S = best_solution[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
