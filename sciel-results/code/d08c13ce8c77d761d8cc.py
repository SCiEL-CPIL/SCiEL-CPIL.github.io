"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = START_TIME + TIME_LIMIT
    
    # 1. Define a base lattice configuration for 12 hexagons
    # We use a 4-4-4 arrangement on a hexagonal grid.
    # Hexagonal grid coordinates (axial): (q, r)
    # Physical coordinates:
    # x = 1.5 * q
    # y = sqrt(3) / 2 * (2*r + q)  (for pointy top? No, flat top usually)
    # Let's use flat top hexagons.
    # Width = 2, Height = sqrt(3).
    # Horizontal spacing = 1.5. Vertical spacing = sqrt(3).
    # Offset for odd columns = sqrt(3)/2.
    
    # Lattice points for 12 hexagons (4 columns, 3 rows? No, 12 = 4x3 or 3x4 or 2x6)
    # Let's try a 3x4 grid (3 columns, 4 rows) or 4x3.
    # A compact cluster is better.
    # Center: (0,0)
    # Neighbors: 6
    # Next ring: 12. Total 19.
    # We need 12.
    # Take the center + 6 neighbors + 5 from the next ring?
    # Or just a rectangular grid of hexes.
    
    # Let's generate a hexagonal grid of 12 points.
    # 4 columns, 3 rows.
    # q in [-1, 0, 1, 2], r in [-1, 0, 1] ?
    # Let's use a simple rectangular grid of hexagons, then optimize.
    # Actually, a "flower" shape is more compact.
    # Center (0,0).
    # 6 around it.
    # 5 more around that.
    
    # Let's just use a grid and let the optimizer fix the shape.
    # Initial guess: 4x3 grid of hexagons.
    # q: -1.5, -0.5, 0.5, 1.5
    # r: -1, 0, 1
    
    # Physical coords for flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    q_coords = np.array([-1.5, -0.5, 0.5, 1.5])
    r_coords = np.array([-1.0, 0.0, 1.0])
    
    init_x = []
    init_y = []
    for q in q_coords:
        for r in r_coords:
            x = 1.5 * q
            y = math.sqrt(3) * (r + q/2.0)
            init_x.append(x)
            init_y.append(y)
            
    init_x = np.array(init_x)
    init_y = np.array(init_y)
    init_theta = np.zeros(N)
    
    # Variables: x, y, theta for each of the 12 hexagons.
    # Plus outer hexagon rotation? No, outer hexagon is fixed orientation usually, or we can rotate it.
    # The problem says "outer hexagon center (x, y), side length and rotation".
    # We can fix outer center at (0,0) and rotation at 0 without loss of generality (by symmetry/translation).
    # We only need to minimize the side length S such that all inner hexagons are inside.
    
    # Objective: Minimize S
    # Constraints:
    # 1. Inner hexagons do not overlap each other.
    # 2. Inner hexagons are inside the outer hexagon of side S.
    
    # This is a non-convex problem. We can use a penalty method or SLSQP with constraints.
    # Given time limits, a penalty method with L-BFGS-B is robust.
    
    # Vertices of a unit hexagon (flat top, centered at origin, angle 0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    def get_hex_vertices(cx, cy, theta):
        angles = theta + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def get_outer_vertices(S, center=(0,0), rotation=0.0):
        angles = rotation + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        return np.column_stack([center[0] + S * np.cos(angles), center[1] + S * np.sin(angles)])
    
    # Point in polygon check (convex polygon)
    # For a convex polygon defined by vertices in CCW order, a point is inside if it is to the left of all edges.
    # Or use cross products.
    
    def point_in_convex_polygon(pt, poly):
        # poly: (6, 2) array
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1) % n]
            # Cross product (p2-p1) x (pt-p1)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            px = pt[0] - p1[0]
            py = pt[1] - p1[1]
            cross = dx * py - dy * px
            if cross < -1e-9: # Inside means same sign for all (assuming CCW)
                return False
        return True
    
    # Distance between two hexagons (non-overlap constraint)
    # This is complex. A simpler penalty:
    # Check if any vertex of hex A is inside hex B, or any vertex of B is inside hex A.
    # This is not sufficient (edge crossing without vertex inside), but for hexagons it's a good start.
    # Better: Compute distance between convex polygons.
    # Since we are minimizing S, we can use a penalty for overlap.
    
    # Let's use a simpler approach:
    # Minimize S + Penalty(Overlap)
    # Overlap penalty: Sum of max(0, depth) for all pairs.
    # Depth of penetration is hard to compute exactly without GJK or similar.
    # Approximation: If vertices are inside, penalty is large.
    
    # Alternative: Use the fact that hexagons are convex.
    # Separating Axis Theorem (SAT).
    # If there is a separating axis, they don't overlap.
    # The axes are the normals to the edges of both hexagons.
    
    def hex_overlap_penalty(v1, v2):
        # v1, v2: (6,2) vertices of two hexagons
        # Returns 0 if no overlap, else penalty (approx depth)
        # Axes: normals of edges of v1 and v2
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            axes.append(normal)
            
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            axes.append(normal)
            
        min_overlap = float('inf')
        
        for axis in axes:
            # Project v1 and v2 onto axis
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap on this axis
            if max1 < min2 or max2 < min1:
                # Separated
                return 0.0
            
            # Overlap amount on this axis
            overlap = min(max1, max2) - max(min1, min2)
            min_overlap = min(min_overlap, overlap)
            
        return min_overlap

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # Unpack params
        # params: [x0, y0, t0, x1, y1, t1, ...]
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # S is not in params. We want to find the minimal S.
        # But S depends on positions.
        # We can iterate:
        # 1. Fix S, check if valid.
        # 2. Binary search on S?
        # Or, define objective as S + Penalty.
        # But S is not a variable in the optimization vector easily if we don't know it.
        
        # Let's change strategy:
        # Optimize positions to minimize the required S.
        # Required S is the maximum distance from center to any vertex of any inner hexagon?
        # No, the inner hexagons must be INSIDE the outer hexagon.
        # The outer hexagon is a convex container.
        # The minimal S is determined by the "support function" of the union of inner hexagons in the direction of the outer hexagon's normals.
        
        # Let's assume the outer hexagon is centered at (0,0) and rotated by 0 (flat top).
        # Its normals are at 0, 60, ..., 300 degrees.
        # For a point p, the distance to the boundary in direction n is S - dot(p, n) ?
        # For a regular hexagon of side S centered at origin, the distance from center to edge in direction n (normal) is S * cos(30) = S * sqrt(3)/2 ?
        # No.
        # Flat top hexagon:
        # Vertices at (S, 0), (S/2, S*sqrt(3)/2), ...
        # The edge from (S,0) to (S/2, S*sqrt(3)/2).
        # Midpoint: (3S/4, S*sqrt(3)/4).
        # Distance from origin to this edge?
        # The apothem (inradius) of a regular hexagon is S * sqrt(3)/2.
        # So the distance from center to any edge is S * sqrt(3)/2.
        
        # So, for a point p to be inside the outer hexagon, we must have:
        # dot(p, n_i) <= S * sqrt(3)/2 for all 6 normal directions n_i.
        # Where n_i are unit vectors at 0, 60, ..., 300 degrees.
        
        # Thus, the minimal S required for a set of points P is:
        # S = max( max_{p in P} dot(p, n_i) ) / (sqrt(3)/2)
        # for all i.
        
        # So we can compute the required S for the current configuration.
        
        normals = np.array([
            [1.0, 0.0],
            [0.5, math.sqrt(3)/2],
            [-0.5, math.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -math.sqrt(3)/2],
            [0.5, -math.sqrt(3)/2]
        ])
        
        max_dot = 0.0
        for i in range(N):
            verts = get_hex_vertices(xs[i], ys[i], thetas[i])
            for j in range(6):
                d = np.dot(verts[j], normals)
                if d > max_dot:
                    max_dot = d
                    
        S = max_dot / (math.sqrt(3)/2)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            v1 = get_hex_vertices(xs[i], ys[i], thetas[i])
            for j in range(i+1, N):
                v2 = get_hex_vertices(xs[j], ys[j], thetas[j])
                pen = hex_overlap_penalty(v1, v2)
                penalty += pen**2
                
        return S + 100.0 * penalty

    # Initial guess
    x0 = np.zeros(N * 3)
    x0[0::3] = init_x
    x0[1::3] = init_y
    x0[2::3] = init_theta
    
    # Optimization
    best_params = x0
    best_score = objective(x0)
    
    # Try a few random restarts or just one L-BFGS-B run
    # L-BFGS-B is good for this.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'maxfun': 10000})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception as e:
        pass
        
    # Extract best configuration
    xs = best_params[0::3]
    ys = best_params[1::3]
    thetas = best_params[2::3]
    
    # Calculate final S
    normals = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    max_dot = 0.0
    for i in range(N):
        verts = get_hex_vertices(xs[i], ys[i], thetas[i])
        for j in range(6):
            d = np.dot(verts[j], normals)
            if d > max_dot:
                max_dot = d
                
    S = max_dot / (math.sqrt(3)/2)
    
    # Construct output
    inner = [[float(xs[i]), float(ys[i]), float(thetas[i] * 180.0 / math.pi)] for i in range(N)]
    
    # If S is too large or invalid, fallback to a known good config (e.g. seed)
    if S > 10.0 or not np.isfinite(S):
        # Fallback: Seed-like config
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        S = 40.0
    else:
        # Center the configuration
        cx = np.mean(xs)
        cy = np.mean(ys)
        for i in range(N):
            inner[i][0] -= cx
            inner[i][1] -= cy
        # Recalculate S? The relative positions didn't change, so S is the same.
        # But the outer hexagon center should be (0,0).
        
    return inner, (0.0, 0.0), float(S), 0.0
# EVOLVE-BLOCK-END
