import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Generate 12 unit hexagon positions in a compact hexagonal cluster
    # Using axial coordinates for hexagonal grid
    # A compact arrangement of 12: think of it as a hexagonal blob
    
    # Positions in hex grid (axial q, r), convert to Cartesian
    # For pointy-top hexagons, axial to cartesian:
    # x = size * (sqrt(3)*q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    # For side=1, size=1
    
    sqrt3 = np.sqrt(3)
    
    # 12 positions in a compact hexagonal cluster
    # Let's use: a central hexagon, 6 around it (dist 2 in hex units... wait)
    # Adjacent hexagons in hex grid are at distance sqrt(3) in Cartesian for side-1 hexagons
    # No wait: for regular hexagons of side 1 tiling, centers are at distance 2*apothem = 2*(sqrt(3)/2) = sqrt(3)
    
    # Let me place 12 hexagons in a hexagonal pattern:
    # Layer 0: 1 (center)
    # Layer 1: 6 (at distance sqrt(3))
    # Layer 2: 5 more to make 12
    # Total: 1+6+5 = 12
    
    # Better: use a known compact shape. Let me use axial coords:
    # (0,0) - center
    # (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1) - 6 neighbors
    # (2,0),(-2,0),(0,2),(0,-2),(1,-2) - 5 more
    # That's 12
    
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, -2)
    ]
    
    # Convert to Cartesian (for pointy-top hexagons with side 1)
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
    
    # Generate all vertices of all 12 unit hexagons
    # A regular hexagon of side 1 centered at (cx, cy) with rotation angle alpha has vertices at:
    # (cx + cos(alpha + k*pi/3), cy + sin(alpha + k*pi/3)) for k=0..5
    
    def get_hex_vertices(cx, cy, alpha):
        angles = alpha + np.arange(6) * np.pi / 3
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    # For each inner hexagon, we can rotate it independently
    # But for simplicity and symmetry, let's try all same rotation first, then optimize
    
    # The key insight: we need to find the minimal regular hexagon (outer) containing all vertices
    # of all inner hexagons. The outer hexagon has center (0,0), side length s, rotation theta.
    
    # For a regular hexagon with circumradius R (= side length) and rotation theta,
    # the support function in direction phi is:
    # h(phi) = R * cos(pi/6 - delta) where delta = ((phi - theta) % (pi/3)) - pi/6
    # Actually: the six vertices are at angles theta + k*pi/3
    # The edges are perpendicular to directions theta + pi/6 + k*pi/3
    # The distance from center to edge in direction phi is R * cos(angle between phi and nearest edge-normal)
    
    # Edge normals are at angles: theta + pi/6 + k*pi/3 for k=0..5
    # For a point at distance d in direction phi, it's inside if:
    # d * cos(angle_diff) <= R * cos(pi/6) for the nearest edge normal
    # i.e., d <= R * cos(pi/6) / cos(angle_diff) where angle_diff is angle from phi to nearest edge normal
    
    def min_outer_side(centers, inner_rotations, theta_outer):
        """Compute minimal outer hexagon side length for given inner rotations and outer rotation."""
        # Get all vertices
        all_vertices = []
        for i, (cx, cy) in enumerate(centers):
            verts = get_hex_vertices(cx, cy, inner_rotations[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # For each vertex, compute the required R
        R_max = 0
        for vx, vy in all_vertices:
            d = np.sqrt(vx**2 + vy**2)
            if d < 1e-10:
                continue
            phi = np.arctan2(vy, vx)
            # Edge normals of outer hexagon
            edge_normals = theta_outer + np.pi/6 + np.arange(6) * np.pi/3
            # Find minimum angular difference to any edge normal
            diffs = np.abs((phi - edge_normals + np.pi) % (2*np.pi) - np.pi)
            min_diff = np.min(diffs)
            # Required R: d = R * cos(min_diff) / cos(pi/6)... 
            # The constraint is: d * cos(min_diff) <= R * cos(pi/6)
            # So R >= d * cos(min_diff) / cos(pi/6)
            R_required = d * np.cos(min_diff) / np.cos(np.pi/6)
            R_max = max(R_max, R_required)
        
        return R_max
    
    # Optimize: inner rotations (all same for simplicity) and outer rotation
    best_score = float('inf')
    best_params = None
    
    # Try various outer rotations
    for theta_outer in np.linspace(0, np.pi/3, 13):
        # Try various inner rotations
        for inner_rot in np.linspace(0, np.pi/6, 7):
            inner_rots = np.full(12, inner_rot)
            s = min_outer_side(centers, inner_rots, theta_outer)
            if s < best_score:
                best_score = s
                best_params = (inner_rot, theta_outer)
        
        if time.time() > deadline - 10:
            break
    
    if best_params is None:
        # Fallback
        return [[centers[i,0], centers[i,1], 0.0] for i in range(12)], (0.0, 0.0), 40.0, 0.0
    
    inner_rot, theta_outer = best_params
    inner_rots = np.full(12, inner_rot)
    
    # Refine with finer search
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        best_local = best_score
        best_local_params = best_params
        for theta_outer in np.linspace(best_params[1] - np.pi/60, best_params[1] + np.pi/60, 25):
            for inner_rot in np.linspace(best_params[0] - np.pi/30, best_params[0] + np.pi/30, 25):
                inner_rots = np.full(12, inner_rot)
                s = min_outer_side(centers, inner_rots, theta_outer)
                if s < best_local:
                    best_local = s
                    best_local_params = (inner_rot, theta_outer)
        best_score = best_local
        best_params = best_local_params
    
    inner_rot, theta_outer = best_params
    
    # Build result
    inner = []
    for i in range(12):
        inner.append([centers[i, 0], centers[i, 1], np.degrees(inner_rot)])
    
    return inner, (0.0, 0.0), best_score, np.degrees(theta_outer)

