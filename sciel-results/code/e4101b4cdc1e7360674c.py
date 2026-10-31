"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    # Rotate by -angle
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local frame, hexagon has vertices at angles pi/6 + k*pi/3
    # Check if point is inside using the 6 half-plane constraints
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The hexagon in local coords: |lx| <= s*sqrt(3)/2 and |ly| <= s*sqrt(3)/2 and constraints on corners
    # Actually, a regular hexagon with flat-top orientation (vertices at 30,90,150,210,270,330 deg from center)
    # has the property that in local coords, the boundaries are:
    #   |ly| <= s*sqrt(3)/2
    #   |lx| <= s*sqrt(3)/2
    #   |lx| + |ly|/sqrt(3) <= s*sqrt(3)/2  (for the slanted sides)
    # Let me use a more robust check: the hexagon with vertices at angles pi/6+k*pi/3
    # is the intersection of 6 half-planes.
    # Vertex k is at angle pi/6 + k*pi/3, distance s from center.
    # The edge between vertex k and k+1 is perpendicular to the bisector direction.
    # Bisector of edge k is at angle pi/6 + (k+0.5)*pi/3 = (2k+1)*pi/6
    # The apothem (distance from center to edge) = s*cos(pi/6) = s*sqrt(3)/2
    # So constraint: dot((lx,ly), (cos(theta_k), sin(theta_k))) <= s*sqrt(3)/2
    # where theta_k = (2k+1)*pi/6 for k=0..5
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        theta = (2 * k + 1) * np.pi / 6
        if lx * np.cos(theta) + ly * np.sin(theta) > apothem:
            return False
    return True

def _hexagon_overlap(h1, h2):
    """Check if two hexagons overlap. h1=(cx,cy,angle), h2=(cx,cy,angle). Both side=1."""
    # Sample points on boundary of h1 and check if any are inside h2
    # Also check if center of h1 is inside h2 and vice versa
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Quick rejection: distance between centers
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    # Max distance from center to any vertex is 1 (side length 1)
    # So if dist > 2, no overlap possible
    if dist > 2.0:
        return False
    
    # Check if center of h1 is inside h2
    if _point_in_hex(cx1, cy1, cx2, cy2, 1.0, a2):
        return True
    # Check if center of h2 is inside h1
    if _point_in_hex(cx2, cy2, cx1, cy1, 1.0, a1):
        return True
    
    # Check boundary intersections: sample points on h1 boundary
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, 1.0, a2):
            return True
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, 1.0, a1):
            return True
    
    return False

def _min_required_side(positions, angles):
    """Compute minimum outer hexagon side length to contain all unit hexagons.
    positions: list of (x,y), angles: list of angle_degrees.
    Outer hexagon is axis-aligned (0 degrees rotation) centered at origin.
    """
    # The outer hexagon (side S, centered at origin, 0 deg rotation) contains
    # a point (px, py) iff for all k=0..5:
    #   px*cos(theta_k) + py*sin(theta_k) <= S*sqrt(3)/2
    # where theta_k = (2k+1)*pi/6
    # So S >= max over all points of max_k (px*cos(theta_k)+py*sin(theta_k)) / (sqrt(3)/2)
    
    max_val = 0.0
    thetas = [(2*k+1)*np.pi/6 for k in range(6)]
    
    for i, (px, py) in enumerate(positions):
        # Get all vertices of inner hexagon i
        verts = _hex_vertices(px, py, 1.0, angles[i])
        for vx, vy in verts:
            for theta in thetas:
                val = vx * np.cos(theta) + vy * np.sin(theta)
                if val > max_val:
                    max_val = val
    
    return max_val / (np.sqrt(3) / 2)

def _check_validity(positions, angles):
    """Check that all hexagons are pairwise disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap((positions[i][0], positions[i][1], angles[i]),
                                (positions[j][0], positions[j][1], angles[j])):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Start with hexagonal grid arrangement
    # Hex grid: rows offset by sqrt(3)/2, columns offset by 1.5 for same row
    # For N hexagons, arrange in roughly hexagonal shape
    # 1, 3, 5, 7, 5, 3, 1 = 25 for 7 rows; but we have 11
    # Let's use a compact arrangement: 3-4-3-1 or similar
    # Actually, let's start with a simple hexagonal cluster
    
    # Generate positions for N hexagons in a hex grid
    # Hex grid coordinates: (col, row) -> (x, y)
    # x = 1.5 * col, y = sqrt(3) * (row + 0.5 * (col % 2))
    # This gives touching hexagons (distance = sqrt(3) between adjacent centers)
    
    sqrt3 = np.sqrt(3)
    
    # Generate hex grid positions for N items
    def hex_grid_positions(n):
        """Generate n positions in a hex grid, roughly centered."""
        positions = []
        # Add center
        positions.append((0.0, 0.0))
        # Add rings
        ring = 1
        while len(positions) < n:
            # Ring of radius r has 6*r hexagons
            for side in range(6):
                for step in range(ring):
                    # Position along this side
                    # Direction from center
                    base_angle = side * np.pi / 3
                    # Step along the side
                    t = step / ring if ring > 1 else 0
                    # Actually, let me use axial coordinates
                    pass
            
            # Use axial coordinates for hex grid
            # In axial coords (q, r), position is:
            # x = 1.5 * q
            # y = sqrt(3) * (r + q/2)
            # Ring r has all (q, r) with max(|q|, |r|, |q+r|) == r
            for q in range(-ring, ring + 1):
                for r in range(-ring, ring + 1):
                    if max(abs(q), abs(r), abs(q + r)) == ring:
                        x = 1.5 * q
                        y = sqrt3 * (r + q / 2.0)
                        if len(positions) < n:
                            positions.append((x, y))
            ring += 1
        
        return positions[:n]
    
    positions = hex_grid_positions(N)
    angles = [0.0] * N
    
    # Initial side length
    current_side = _min_required_side(positions, angles)
    best_side = current_side
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    # Verify validity
    if not _check_validity(positions, angles):
        # Fallback: linear arrangement
        positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        angles = [0.0] * N
        current_side = _min_required_side(positions, angles)
        best_side = current_side
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
    
    # Optimize: try to reduce side length by adjusting positions
    # Use a simple local search: perturb positions and check if valid
    # and if it reduces the required side
    
    def _objective(x):
        """Objective: minimize max distance from origin (proxy for side length)."""
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        ang = [x[2*i+2] for i in range(N)]
        # Check validity
        for i in range(N):
            for j in range(i+1, N):
                if _hexagon_overlap((pos[i][0], pos[i][1], ang[i]),
                                    (pos[j][0], pos[j][1], ang[j])):
                    return 1e6
        side = _min_required_side(pos, ang)
        return side
    
    # Local search with random perturbations
    np.random.seed(42)
    x0 = []
    for i in range(N):
        x0.extend([positions[i][0], positions[i][1], angles[i]])
    
    x_best = x0[:]
    f_best = _objective(x_best)
    
    # Simple random restart local search
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Start from best or random perturbation
        if restart == 0:
            x = x_best[:]
        else:
            x = x_best + np.random.randn(3*N) * 0.5
        
        for step in range(200):
            if time.time() > deadline - 5.0:
                    break
            
            # Random perturbation
            dx = np.random.randn(3*N) * 0.1
            x_new = x + dx
            
            f_new = _objective(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new[:]
                x = x_new[:]
            elif f_new < 1e6:
                # Accept with some probability (simulated annealing-like)
                if np.random.random() < 0.1:
                    x = x_new[:]
    
    # Extract best solution
    best_positions = [(x_best[2*i], x_best[2*i+1]) for i in range(N)]
    best_angles = [x_best[2*i+2] for i in range(N)]
    best_side = _min_required_side(best_positions, best_angles)
    
    # Verify final validity
    if not _check_validity(best_positions, best_angles):
        # Use the hex grid fallback
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
        best_side = _min_required_side(best_positions, best_angles)
    
    # Add tiny epsilon to ensure strict containment
    best_side *= (1.0 + 1e-6)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
