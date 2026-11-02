"""Optimized hexagon packing using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import random
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Returns:
        inner: list of [x, y, angle_degrees] for each inner hexagon
        center: (x, y) of outer hexagon center
        side: outer hexagon side length
        angle: outer hexagon rotation (degrees)
    """
    if N <= 0:
        return [], (0.0, 0.0), 0.0, 0.0
    
    if N == 1:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
    
    # For a regular hexagon of side length s, the vertices at rotation 0 are:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The distance from center to each vertex is s.
    # The distance from center to each edge (apothem) is s*sqrt(3)/2.
    
    def hexagon_vertices(cx, cy, s, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at (cx, cy) with side s and rotation angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + s * math.cos(theta)
            y = cy + s * math.sin(theta)
            verts.append((x, y))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, s, angle_deg=0.0):
        """Check if point (px, py) is inside or on the boundary of hexagon."""
        # Transform point to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with side s, in local coords with vertex at angle 0:
        # The hexagon is defined by |lx| <= s and |ly| <= s*sqrt(3)/2 and |lx + ly/sqrt(3)| <= s and |lx - ly/sqrt(3)| <= s
        # Actually, let's use the half-plane definition.
        # The six edges of a regular hexagon (vertex at 0 degrees) are:
        # Edge 1: from (s,0) to (s/2, s*sqrt(3)/2): line x + y/sqrt(3) = s
        # Edge 2: from (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2): line y = s*sqrt(3)/2
        # Edge 3: from (-s/2, s*sqrt(3)/2) to (-s, 0): line -x + y/sqrt(3) = s
        # Edge 4: from (-s, 0) to (-s/2, -s*sqrt(3)/2): line x + y/sqrt(3) = -s
        # Edge 5: from (-s/2, -s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2): line y = -s*sqrt(3)/2
        # Edge 6: from (s/2, -s*sqrt(3)/2) to (s, 0): line -x + y/sqrt(3) = -s
        
        sqrt3 = math.sqrt(3)
        constraints = [
            lx + ly / sqrt3 <= s,
            ly <= s * sqrt3 / 2,
            -lx + ly / sqrt3 <= s,
            lx + ly / sqrt3 >= -s,
            ly >= -s * sqrt3 / 2,
            -lx + ly / sqrt3 >= -s
        ]
        return all(c for c in constraints)
    
    def hexagon_overlap_area_approx(v1, v2, s1, s2, a1, a2):
        """Approximate overlap by checking if centers are too close.
        Two unit hexagons don't overlap if their centers are at least 2 apart (for same orientation)
        or at least some minimum distance depending on relative rotation.
        The minimum distance between centers for non-overlapping hexagons of side 1 is 2 (when aligned).
        For different rotations, the minimum distance can be slightly less, but 2 is a safe lower bound
        only if we're being conservative. Actually, the minimum center-to-center distance for two
        non-overlapping regular hexagons of side 1 is 2 (when they share an edge).
        """
        # The inradius of a unit hexagon is sqrt(3)/2.
        # Two hexagons can be as close as 2 * inradius = sqrt(3) if they touch edge-to-edge?
        # No. Let me think again.
        # A regular hexagon of side 1 has circumradius 1 and inradius sqrt(3)/2.
        # If two hexagons share an edge, the distance between centers is 2 * inradius = sqrt(3) ≈ 1.732.
        # Wait, that's not right either. If two regular hexagons of side 1 share a full edge,
        # the centers are separated by 2 * (sqrt(3)/2) = sqrt(3).
        # But they could also be rotated such that a vertex touches an edge, etc.
        # The minimum distance for non-overlap is actually sqrt(3) when they share an edge.
        # Let's use sqrt(3) as the minimum center distance.
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist < math.sqrt(3) - 1e-9
    
    def outer_constraint_violation(pos, angle, outer_s, outer_cx, outer_cy, outer_angle):
        """Compute how much a hexagon at pos with angle violates the outer boundary.
        Returns the maximum distance by which any vertex of the inner hexagon
        lies outside the outer hexagon. 0 means all vertices are inside or on boundary.
        """
        inner_verts = hexagon_vertices(pos[0], pos[1], 1.0, angle)
        max_violation = 0.0
        for vx, vy in inner_verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
                # Compute distance from point to outer hexagon boundary
                # This is complex, so let's use a simpler approach:
                # Check how far outside it is by scaling.
                # For now, just count it as a violation of magnitude 1.0
                max_violation = max(max_violation, 1.0)
        return max_violation
    
    def objective(params):
        """Objective function: minimize outer side length subject to constraints.
        We'll use a penalty method.
        params: [outer_s, outer_cx, outer_cy, outer_angle, x1, y1, a1, x2, y2, a2, ...]
        """
        outer_s = params[0]
        outer_cx = params[1]
        outer_cy = params[2]
        outer_angle = params[3]
        
        if outer_s < 1.0:
            return 1e10
        
        positions = []
        angles = []
        for i in range(N):
            x = params[4 + i*3]
            y = params[4 + i*3 + 1]
            a = params[4 + i*3 + 2]
            positions.append((x, y))
            angles.append(a)
        
        penalty = 0.0
        
        # Check outer boundary constraints
        for i in range(N):
            v = outer_constraint_violation(positions[i], angles[i], outer_s, outer_cx, outer_cy, outer_angle)
            penalty += v * v
        
        # Check pairwise overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                if hexagon_overlap_area_approx(positions[i], positions[j], 1.0, 1.0, angles[i], angles[j]):
                    penalty += 100.0
        
        return outer_s + penalty * 1000.0
    
    # Multi-start optimization
    best_solution = None
    best_score = float('inf')
    
    # Initial guess: arrange hexagons in a grid-like pattern
    # For N hexagons, a hexagonal arrangement is good.
    # Let's try several random initializations.
    
    num_starts = min(5, max(1, 20 - N))
    
    for start in range(num_starts):
        if start == 0:
            # Structured initial guess: hexagonal lattice
            # Place hexagons in a hexagonal pattern
            positions_init = []
            angles_init = []
            # For N hexagons, use a hexagonal close-packed arrangement
            # Row 0: 1 hex
            # Row 1: 2 hexes
            # etc.
            row = 0
            count = 0
            while count < N:
                num_in_row = row + 1 if row < 3 else max(1, 6 - row)
                for k in range(min(num_in_row, N - count)):
                    if count >= N:
                        break
                    # Hexagonal lattice spacing
                    x = (k - row/2) * 1.8  # slightly more than sqrt(3) to avoid overlap
                    y = row * 1.5
                    positions_init.append((x, y))
                    angles_init.append(0.0)
                    count += 1
                row += 1
                if row > 10:
                    break
            
            # Pad if needed
            while len(positions_init) < N:
                positions_init.append((0.0, 0.0))
                angles_init.append(0.0)
        else:
            # Random initial guess
            positions_init = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(N)]
            angles_init = [random.uniform(0, 60) for _ in range(N)]
        
        # Estimate initial outer side length
        max_dist = 0
        for pos in positions_init:
            max_dist = max(max_dist, abs(pos[0]), abs(pos[1]))
        outer_s_init = max_dist + 2.0
        
        # Build initial params
        params_init = [outer_s_init, 0.0, 0.0, 0.0]
        for i in range(N):
            params_init.extend([positions_init[i][0], positions_init[i][1], angles_init[i]])
        
        # Optimize
        try:
            result = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 10000}
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception:
            continue
    
    if best_solution is None:
        # Fallback: simple grid
        inner = []
        for i in range(N):
            inner.append([float(i) * 2.0, 0.0, 0.0])
        return inner, (0.0, 0.0), float(N) * 2.0 + 1.0, 0.0
    
    # Extract solution
    outer_s = max(1.0, best_solution[0])
    outer_cx = best_solution[1]
    outer_cy = best_solution[2]
    outer_angle = best_solution[3]
    
    inner = []
    for i in range(N):
        x = best_solution[4 + i*3]
        y = best_solution[4 + i*3 + 1]
        a = best_solution[4 + i*3 + 2]
        inner.append([float(x), float(y), float(a)])
    
    # Verify and adjust if necessary
    # If the solution is invalid, use a conservative fallback
    valid = True
    for i in range(N):
        if outer_constraint_violation((inner[i][0], inner[i][1]), inner[i][2], outer_s, outer_cx, outer_cy, outer_angle) > 0:
            valid = False
            break
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_overlap_area_approx((inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]), 1.0, 1.0, inner[i][2], inner[j][2]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Conservative fallback: place hexagons in a line with spacing sqrt(3)
        inner = []
        for i in range(N):
            inner.append([float(i) * math.sqrt(3), 0.0, 0.0])
        outer_s = float(N) * math.sqrt(3) + 1.0
        outer_cx = (N-1) * math.sqrt(3) / 2.0
        outer_cy = 0.0
        outer_angle = 0.0
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_s), float(outer_angle)
# EVOLVE-BLOCK-END
