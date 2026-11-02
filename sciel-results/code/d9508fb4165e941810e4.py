"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # vertices of a regular hexagon of side length 1 (radius 1)
    angles = np.pi / 6 + np.arange(6) * np.pi / 3
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    # rotate around center
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return np.column_stack([xr, yr])

def _outer_boundary_distance(points, center, side, angle_deg):
    """
    For each point, compute the distance to the interior of the outer hexagon.
    Returns a vector of signed distances: positive if inside, negative if outside.
    The outer hexagon has center, side length s, and rotation angle_deg.
    A point is inside iff it satisfies 6 half-space constraints.
    For a regular hexagon with side s, the apothem (inradius) is s * sqrt(3) / 2.
    The 6 normals point from center to the midpoints of the sides.
    """
    a = np.radians(angle_deg)
    apothem = side * np.sqrt(3) / 2
    # normals at angles 0, 60, 120, 180, 240, 300 degrees rotated by angle_deg
    normal_angles = a + np.arange(6) * np.pi / 3
    # The boundary is at distance apothem from center along each normal direction
    # Point p is inside iff (p - center) dot n_i <= apothem for all i
    # where n_i are unit normals pointing outward
    # Actually, let's define inward normals: point is inside iff (p - center) dot (-n_i) <= apothem
    # Let's use: for each normal direction theta, the constraint is:
    # (p_x - cx) * cos(theta) + (p_y - cy) * sin(theta) <= apothem
    # where theta are the directions of the outward normals (perpendicular to sides)
    # For a regular hexagon, the outward normals are at angles 0, 60, ..., 300 (for a flat-top or pointy-top)
    # Let's just compute the 6 support function values
    
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    # Rotate points so the hexagon is axis-aligned
    ca, sa = np.cos(a), np.sin(a)
    dxr = ca * dx + sa * dy
    dyr = -sa * dx + ca * dy
    
    # For a regular hexagon with apothem r, the 6 constraints are:
    # |dxr| <= r, |dyr| <= r, |dxr + sqrt(3)*dyr|/2 <= r, etc.
    # More precisely, the 6 outward normals (for a hexagon with flat top, i.e., one side horizontal)
    # are at angles 0, 60, 120, 180, 240, 300.
    # The constraints are: dxr*cos(0) + dyr*sin(0) <= r  =>  dxr <= r
    #                    dxr*cos(60) + dyr*sin(60) <= r  =>  dxr/2 + sqrt(3)*dyr/2 <= r
    #                    ...
    
    constraints = np.zeros((points.shape[0], 6))
    constraints[:, 0] = dxr
    constraints[:, 1] = -dxr
    constraints[:, 2] = dxr * 0.5 + dyr * np.sqrt(3) * 0.5
    constraints[:, 3] = -dxr * 0.5 + dyr * np.sqrt(3) * 0.5
    constraints[:, 4] = -dxr * 0.5 - dyr * np.sqrt(3) * 0.5
    constraints[:, 5] = dxr * 0.5 - dyr * np.sqrt(3) * 0.5
    
    # The max constraint value gives the "distance" in terms of violation
    # Actually, the minimum over the 6 of (apothem - constraint_i) gives the margin
    # If all constraints_i <= apothem, the point is inside.
    # The distance to the boundary (in the L_infinity sense for this polytope) is min_i (apothem - constraint_i)
    margins = apothem - constraints  # each column: apothem - constraint
    return margins  # shape (n_points, 6), positive means inside with that margin

def _min_margin(points, center, side, angle_deg):
    """Minimum margin over all points and all 6 sides."""
    margins = _outer_boundary_distance(points, center, side, angle_deg)
    return margins.min()

def construct():
    start_time = time.time()
    time_limit = 145.0  # safety margin
    
    # Initial guess: place hexagons in a grid-like pattern
    # For N hexagons, arrange in rows
    # A hexagonal lattice has spacing sqrt(3) in one direction and 3/2 in the other for adjacent rows
    
    # Let's try a simple packing: arrange in a honeycomb pattern
    # For N hexagons, we can compute a reasonable initial configuration
    
    # Initial positions: use a hexagonal lattice
    # The minimal enclosing hexagon for N unit hexagons in a hexagonal packing
    # has side length related to sqrt(N)
    
    # Let's place them in a pattern that minimizes the bounding hexagon
    # For now, use a simple grid and let the optimizer work
    
    # Compute initial positions
    # Arrange in rows: for N items, use roughly sqrt(N) columns
    n_cols = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    positions = []
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        # Hexagonal lattice offset
        x = col * 1.5
        y = row * np.sqrt(3)
        if row % 2 == 1:
            x += 0.75
        positions.append([x, y, 0.0])
    
    # Center the configuration
    if positions:
        cx_mean = np.mean([p[0] for p in positions])
        cy_mean = np.mean([p[1] for p in positions])
        for p in positions:
            p[0] -= cx_mean
            p[1] -= cy_mean
    
    # Initial outer side length: estimate from bounding box
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    # The apothem needed to contain the bounding box
    max_x = max(abs(x) for x in xs) + 1  # +1 for hexagon radius
    max_y = max(abs(y) for y in ys) + 1
    # For a hexagon, the apothem must be at least max over all directions
    # Simple estimate: side = 2 * max(max_x, max_y) / sqrt(3)
    initial_side = 2.0 * max(max_x, max_y) / np.sqrt(3) * 1.5
    
    # Objective: minimize outer side length
    # Decision variables: x, y, angle for each hexagon, plus outer side length
    # We can fix the outer center at origin and outer angle at 0 for simplicity
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10
        
        # Unpack parameters
        # First N*3 are (x, y, angle) for each inner hexagon
        # Last 1 is the outer side length
        inner_params = params[:N * 3].reshape(N, 3)
        side = params[N * 3]
        
        if side < 1.0:
            return 1e10
        
        # Compute all vertices
        all_vertices = []
        for i in range(N):
            x, y, ang = inner_params[i]
            verts = _hex_vertices(x, y, ang)
            all_vertices.append(verts)
        
        all_verts = np.vstack(all_vertices)
        
        # Check outer boundary constraint
        min_margin = _min_margin(all_verts, (0.0, 0.0), side, 0.0)
        
        # If any vertex is outside, penalize heavily
        if min_margin < 0:
            boundary_penalty = 1000.0 * min_margin ** 2
        else:
            boundary_penalty = 0.0
        
        # Check pairwise non-overlap
        # For two convex polygons, they don't overlap iff there exists a separating axis
        # For regular hexagons, we can check if the minimum distance between them is >= 0
        # A simpler approach: check if any vertex of one is inside the other, or if edges intersect
        # For efficiency, use a simpler distance-based penalty
        # The minimum distance between two hexagons can be approximated by the minimum distance
        # between their vertices and edges
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Check distance between hexagons i and j
                # A simple check: minimum distance between vertices
                vi = _hex_vertices(*inner_params[i])
                vj = _hex_vertices(*inner_params[j])
                
                # Compute minimum distance between the two sets of vertices
                # This is O(36) which is fine for small N
                min_dist = np.inf
                for p in vi:
                    d = np.min(np.sqrt(np.sum((vj - p) ** 2, axis=1)))
                    min_dist = min(min_dist, d)
                for p in vj:
                    d = np.min(np.sqrt(np.sum((vi - p) ** 2, axis=1)))
                    min_dist = min(min_dist, d)
                
                # For two unit hexagons, they overlap if the minimum distance between
                # their boundaries is less than some threshold
                # Actually, two hexagons of side 1 have a minimum distance of 0 when touching
                # The center-to-center distance for non-overlapping hexagons depends on orientation
                # For simplicity, if min_dist < 0.05, they likely overlap
                # But this is not exact. Let's use a more robust check.
                
                # Check if any vertex of vi is inside vj or vice versa
                # A point is inside a hexagon if it's on the correct side of all 6 edges
                # This is more expensive but more accurate
                
                # For now, use the vertex distance as a proxy
                if min_dist < 0.1:
                    overlap_penalty += (0.1 - min_dist) ** 2
        
        return side + boundary_penalty + 10.0 * overlap_penalty
    
    # Initial parameter vector
    initial_params = np.array([p for pos in positions for p in pos] + [initial_side])
    
    # Optimize
    best_params = initial_params.copy()
    best_obj = objective(initial_params)
    
    # Multi-start
    np.random.seed(42)
    n_starts = 5
    for s in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
        
        if s == 0:
            params = initial_params.copy()
        else:
            # Perturb initial parameters
            params = initial_params + np.random.randn(len(initial_params)) * 0.1
        
        try:
            result = minimize(objective, params, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-6})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final configuration
    elapsed = time.time() - start_time
    inner_params = best_params[:N * 3].reshape(N, 3)
    side = best_params[N * 3]
    
    # Ensure side is at least 1.0
    side = max(side, 1.0)
    
    inner = [[float(x), float(y), float(ang)] for x, y, ang in inner_params]
    
    return inner, (0.0, 0.0), float(side), 0.0
# EVOLVE-BLOCK-END
