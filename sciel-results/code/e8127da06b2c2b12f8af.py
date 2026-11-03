"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place N unit hexagons on a hexagonal lattice inside a minimal outer hexagon
    # Hexagonal lattice: centers at positions based on axial coordinates
    # Distance between adjacent centers: sqrt(3) for edge-parallel hexagons
    
    def hexagon_vertices(center, side, rotation_deg):
        """Return vertices of a regular hexagon."""
        cx, cy = center
        rot = np.radians(rotation_deg)
        verts = []
        for i in range(6):
            angle = rot + i * np.pi / 3
            verts.append((cx + side * np.cos(angle), cy + side * np.sin(angle)))
        return np.array(verts)
    
    def point_in_hexagon(point, center, side, rotation_deg):
        """Check if a point is inside a regular hexagon."""
        cx, cy = center
        px, py = point
        rot = np.radians(rotation_deg)
        # Transform to hexagon's local frame
        dx, dy = px - cx, py - cy
        cos_r, sin_r = np.cos(-rot), np.sin(-rot)
        lx = dx * cos_r - dy * sin_r
        ly = dx * sin_r + dy * cos_r
        # Hexagon with flat top: inradius = side * sqrt(3)/2
        inradius = side * np.sqrt(3) / 2
        # Check against each edge
        for i in range(6):
            angle = i * np.pi / 3
            # Normal direction
            nx, ny = np.cos(angle + np.pi/6), np.sin(angle + np.pi/6)
            # Project point onto normal
            proj = lx * nx + ly * ny
            if proj > inradius:
                return False
        return True
    
    def hexagons_disjoint(c1, a1, c2, a2, side=1.0):
        """Check if two unit hexagons are disjoint (non-overlapping)."""
        # Two hexagons of side 1 with centers c1, c2 and rotations a1, a2
        # They are disjoint if the distance between centers is >= 2*inradius = sqrt(3)
        # when they have the same orientation. For different orientations, need more care.
        # For simplicity, use the same orientation for all inner hexagons.
        d = np.linalg.norm(np.array(c1) - np.array(c2))
        return d >= np.sqrt(3) - 1e-9
    
    def all_hexagons_disjoint(positions, angles, side=1.0):
        """Check all pairs are disjoint."""
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                if not hexagons_disjoint(positions[i], angles[i], positions[j], angles[j], side):
                    return False
        return True
    
    def all_in_outer(positions, angles, outer_center, outer_side, outer_rot, side=1.0):
        """Check all inner hexagons are inside the outer hexagon."""
        for i in range(len(positions)):
            # Check all vertices of inner hexagon i are inside outer
            inner_verts = hexagon_vertices(positions[i], side, angles[i])
            for v in inner_verts:
                if not point_in_hexagon(v, outer_center, outer_side, outer_rot):
                    return False
        return True
    
    def compute_outer_side(positions, angles, outer_center, outer_rot, side=1.0):
        """Compute the minimum outer side length needed to contain all inner hexagons."""
        # For each inner hexagon, compute the maximum "extent" in each of the 6 directions
        # of the outer hexagon
        outer_rot_rad = np.radians(outer_rot)
        inradius = side * np.sqrt(3) / 2
        
        max_proj = 0
        for pos in positions:
            inner_verts = hexagon_vertices(pos, side, angles[0])  # all same angle
            for v in inner_verts:
                dx, dy = v[0] - outer_center[0], v[1] - outer_center[1]
                # Project onto each of the 6 normal directions
                for k in range(6):
                    angle = outer_rot_rad + k * np.pi / 3 + np.pi/6
                    proj = dx * np.cos(angle) + dy * np.sin(angle)
                    max_proj = max(max_proj, proj)
        
        # The inradius of the outer hex must be at least max_proj
        outer_inradius = max_proj
        outer_side = outer_inradius * 2 / np.sqrt(3)
        return outer_side
    
    # Generate hexagonal lattice positions
    # Axial coordinates: (q, r) -> (x, y) = (sqrt(3)*q + sqrt(3)/2*r, 3/2*r)
    # For hexagons of side 1, the lattice spacing is sqrt(3) in the "flat" direction
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_xy(q, r):
        x = sqrt3 * q + sqrt3/2 * r
        y = 3/2 * r
        return (x, y)
    
    # Generate candidate positions on hexagonal lattice
    # Try different ranges of axial coordinates
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different outer rotations
    for outer_rot in [0.0, 30.0]:
        # Generate lattice points
        # For N=11, we need a compact cluster
        # Try hexagonal clusters of various sizes
        for radius in range(0, 5):
            # Generate all lattice points within hexagonal distance <= radius
            points = []
            for q in range(-radius, radius+1):
                for r in range(max(-radius, -q-radius), min(radius, -q+radius)+1):
                    points.append((q, r))
            
            if len(points) < N:
                continue
            
            # Try all subsets of size N (for small N this is feasible)
            # For larger N, use heuristics
            if len(points) <= 25:
                # Try all combinations
                best_local = float('inf')
                best_local_positions = None
                for combo in itertools.combinations(range(len(points)), N):
                    positions = [axial_to_xy(points[i][0], points[i][1]) for i in combo]
                    angles = [0.0] * N
                    
                    # Center the cluster
                    cx = np.mean([p[0] for p in positions])
                    cy = np.mean([p[1] for p in positions])
                    centered_positions = [(p[0]-cx, p[1]-cy) for p in positions]
                    
                    # Compute required outer side
                    outer_center = (0.0, 0.0)
                    side = compute_outer_side(centered_positions, angles, outer_center, outer_rot)
                    
                    if side < best_local:
                        best_local = side
                        best_local_positions = centered_positions
                
                if best_local < best_side:
                    best_side = best_local
                    best_inner = [[p[0], p[1], 0.0] for p in best_local_positions]
                    best_center = (0.0, 0.0)
                    best_angle = outer_rot
            else:
                # Use a heuristic: take the most central points
                points_with_dist = [(p, abs(p[0]) + abs(p[1]) + abs(p[0]+p[1])) for p in points]
                points_with_dist.sort(key=lambda x: x[1])
                selected = [p for p, _ in points_with_dist[:N]]
                positions = [axial_to_xy(q, r) for q, r in selected]
                angles = [0.0] * N
                
                cx = np.mean([p[0] for p in positions])
                cy = np.mean([p[1] for p in positions])
                centered_positions = [(p[0]-cx, p[1]-cy) for p in positions]
                
                outer_center = (0.0, 0.0)
                side = compute_outer_side(centered_positions, angles, outer_center, outer_rot)
                
                if side < best_side:
                    best_side = side
                    best_inner = [[p[0], p[1], 0.0] for p in centered_positions]
                    best_center = (0.0, 0.0)
                    best_angle = outer_rot
    
    # Fallback: simple linear arrangement
    if best_inner is None:
        best_inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        best_side = (N-1) * sqrt3 + 2
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END

