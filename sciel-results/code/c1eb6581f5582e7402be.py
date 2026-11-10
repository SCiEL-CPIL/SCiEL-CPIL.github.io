import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
sqrt3 = math.sqrt(3.0)
EPS = 1e-10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Helper functions for hexagon geometry ---

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Vertices of a regular hexagon with given center, angle (degrees), and side length."""
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + side*math.cos(a), cy + side*math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        """Ray-casting point-in-polygon test."""
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        """Check if two line segments intersect (excluding touching endpoints)."""
        def ccw(A, B, C):
            val = (C[1]-A[1]) * (B[0]-A[0]) - (B[1]-A[1]) * (C[0]-A[0])
            return val > EPS
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2, tol=1e-4):
        """Check if two convex polygons are disjoint (no overlap, touching allowed)."""
        # Check if any vertex of one is inside the other
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        # Check if any edges intersect
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1) % n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1) % n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def min_distance_between_polygons(v1, v2):
        """Minimum distance between two disjoint convex polygons."""
        min_dist = float('inf')
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i+1) % len(v1)]
            for j in range(len(v2)):
                p3 = v2[j]
                p4 = v2[(j+1) % len(v2)]
                # Point-segment distances
                d = _point_segment_dist(p1, p3, p4)
                min_dist = min(min_dist, d)
                d = _point_segment_dist(p3, p1, p2)
                min_dist = min(min_dist, d)
        return min_dist

    def _point_segment_dist(p, a, b):
        """Distance from point p to segment ab."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if dx == 0 and dy == 0:
            return math.hypot(p[0]-a[0], p[1]-a[1])
        t = ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        proj_x = a[0] + t*dx
        proj_y = a[1] + t*dy
        return math.hypot(p[0]-proj_x, p[1]-proj_y)

    # --- Compute minimum outer side length for a set of inner hexagons ---

    def compute_min_L(inner_positions, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        Compute the minimum side length L of a regular hexagon (centered at outer_center,
        rotated by outer_angle degrees) that contains all inner hexagons.
        inner_positions: list of (x, y, angle_deg) for each inner hexagon (side=1)
        """
        # For a regular hexagon with side L centered at origin with "pointy-top" orientation
        # (vertex at top), the support function in direction theta is L * max over 6 directions
        # The 6 outward normal directions for a pointy-top hex are at 30, 90, 150, 210, 270, 330 degrees
        # Actually, for a flat-top hex (side on top), normals are at 0, 60, 120, 180, 240, 300
        # Let's use: for a regular hexagon with side s, the distance from center to each side (apothem) = s*sqrt(3)/2
        # The support in the 6 normal directions must be >= apothem = s*sqrt(3)/2
        # So s = 2 * max_support / sqrt(3) where max_support is the max projection of any inner vertex
        # onto the 6 outward normal directions of the outer hexagon.
        
        ox, oy = outer_center
        o_ang = math.radians(outer_angle)
        
        # Normal directions for the outer hexagon
        # For a regular hexagon with a vertex at angle o_ang, the 6 side normals are at
        # o_ang + 30 + 60*k for k=0..5 (these are perpendicular to the sides)
        normal_angles = [o_ang + math.radians(30 + 60*k) for k in range(6)]
        
        max_proj = 0.0
        for (ix, iy, i_ang_deg) in inner_positions:
            verts = get_hex_vertices(ix, iy, i_ang_deg, side=1.0)
            for (vx, vy) in verts:
                # Relative to outer center
                rx = vx - ox
                ry = vy - oy
                for na in normal_angles:
                    proj = rx * math.cos(na) + ry * math.sin(na)
                    if proj > max_proj:
                        max_proj = proj
        
        if max_proj <= 0:
            return 1.0
        L = 2.0 * max_proj / sqrt3
        return L

    def is_valid_packing(inner_positions, outer_L, outer_center=(0.0, 0.0), outer_angle=0.0, tol=1e-3):
        """Check if all inner hexagons are disjoint and inside the outer hexagon."""
        verts_list = [get_hex_vertices(x, y, a, side=1.0) for (x, y, a) in inner_positions]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    return False
        
        # Check containment in outer hexagon
        outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_angle, side=outer_L)
        for i in range(N):
            for v in verts_list[i]:
                if not point_in_polygon(v, outer_verts):
                    return False
            # Also check that vertices are not too close to boundary
        return True

    # --- Objective functions for optimization ---

    def objective(params, outer_angle=0.0):
        """Minimize outer L with penalty for overlaps."""
        if time.time() > deadline:
            return 1e10
        inner_positions = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        L = compute_min_L(inner_positions, (0.0, 0.0), outer_angle)
        
        # Penalty for overlaps
        verts_list = [get_hex_vertices(x, y, a, side=1.0) for (x, y, a) in inner_positions]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Quick center-distance check
                d = math.hypot(params[3*i]-params[3*j], params[3*i+1]-params[3*j+1])
                if d < 2.0:  # Hexagons of side 1 have diameter 2
                    # More detailed check
                    d_min = min_distance_between_polygons(verts_list[i], verts_list[j])
                    if d_min < 1e-4:
                        penalty += (1e-4 - d_min)**2
                    else:
                        # Small penalty for very close hexagons to encourage separation
                        if d_min < 0.01:
                            penalty += (0.01 - d_min)**2 * 0.1
        return L + 100.0 * penalty

    def objective_strict(params, outer_angle=0.0):
        """Same as objective but with heavier penalty."""
        if time.time() > deadline:
            return 1e10
        inner_positions = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        L = compute_min_L(inner_positions, (0.0, 0.0), outer_angle)
        
        verts_list = [get_hex_vertices(x, y, a, side=1.0) for (x, y, a) in inner_positions]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(params[3*i]-params[3*j], params[3*i+1]-params[3*j+1])
                if d < 2.0:
                    d_min = min_distance_between_polygons(verts_list[i], verts_list[j])
                    if d_min < 1e-4:
                        penalty += (1e-4 - d_min)**2 * 10.0
                    else:
                        if d_min < 0.01:
                            penalty += (0.01 - d_min)**2
        return L + 1000.0 * penalty

    # --- Generate initial configurations ---

    def generate_grid_init():
        """Hexagonal grid arrangement of 12 hexagons."""
        centers = []
        # Use axial coordinates for hexagonal grid
        # For a hexagon of side 1, grid spacing in x = sqrt(3), in y = 1.5
        for q in range(-3, 4):
            for r in range(-3, 4):
                if abs(q + r) <= 3:
                    x = sqrt3 * (q + r / 2.0)
                    y = 1.5 * r
                    centers.append((x, y, 0.0))
        # Sort by distance from origin and take 12
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        return centers[:12]

    def generate_ring_init(r=2.2):
        """12 hexagons in a ring."""
        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            a = angle * 180 / math.pi
            positions.append((x, y, a))
        return positions

    def generate_random_init(seed=42):
        """Random initialization."""
        np.random.seed(seed)
        positions = []
        for _ in range(N):
            x = np.random.uniform(-3.0, 3.0)
            y = np.random.uniform(-3.0, 3.0)
            a = np.random.uniform(0, 360)
            positions.append((x, y, a))
        return positions

    # --- Main optimization loop ---

    best_L = float('inf')
    best_params = None
    best_outer_angle = 0.0

    # Try multiple outer angles
    outer_angles = [0.0, 30.0, 15.0, 45.0]

    for oa in outer_angles:
        if time.time() > deadline - 20.0:
            break

        # Phase 1: Differential Evolution (Global)
        bounds = [(-3.5, 3.5), (-3.5, 3.5), (0.0, 360.0)] * N
        x0 = None

        # Try multiple initializations for DE
        inits = [
            [p[0] for p in generate_grid_init()] + [p[1] for p in generate_grid_init()] + [p[2] for p in generate_grid_init()],
            [p[0] for p in generate_ring_init(2.2)] + [p[1] for p in generate_ring_init(2.2)] + [p[2] for p in generate_ring_init(2.2)],
        ]

        # Flatten inits properly
        inits_flat = []
        for pos_list in [generate_grid_init(), generate_ring_init(2.2)]:
            flat = []
            for p in pos_list:
                flat.extend([p[0], p[1], p[2]])
            inits_flat.append(flat)

        for init_params in inits_flat:
            if time.time() > deadline - 30.0:
                break
            try:
                res = differential_evolution(
                    objective, bounds,
                    x0=init_params,
                    maxiter=80, tol=1e-5,
                    popsize=12, seed=42, polish=False,
                    args=(oa,)
                )
                if res.fun < best_L and res.fun < 1e5:
                    L_candidate = compute_min_L(
                        [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)],
                        (0.0, 0.0), oa
                    )
                    # Verify validity
                    inner_pos = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                    if is_valid_packing(inner_pos, L_candidate, (0.0, 0.0), oa):
                        best_L = L_candidate
                        best_params = res.x.copy()
                        best_outer_angle = oa
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish
        if best_params is not None and time.time() < deadline - 15.0:
            try:
                res = minimize(
                    objective_strict, best_params,
                    method='Nelder-Mead',
                    args=(best_outer_angle,),
                    options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < 1e5:
                    inner_pos = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                    L_candidate = compute_min_L(inner_pos, (0.0, 0.0), best_outer_angle)
                    if is_valid_packing(inner_pos, L_candidate, (0.0, 0.0), best_outer_angle):
                        if L_candidate < best_L:
                            best_L = L_candidate
                            best_params = res.x.copy()
            except Exception:
                pass

        # Phase 3: L-BFGS-B final polish
        if best_params is not None and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    objective_strict, best_params,
                    method='L-BFGS-B',
                    args=(best_outer_angle,),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e5:
                    inner_pos = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                    L_candidate = compute_min_L(inner_pos, (0.0, 0.0), best_outer_angle)
                    if is_valid_packing(inner_pos, L_candidate, (0.0, 0.0), best_outer_angle):
                        if L_candidate < best_L:
                            best_L = L_candidate
                            best_params = res.x.copy()
            except Exception:
                pass

    # If optimization failed, use the grid fallback
    if best_params is None or not math.isfinite(best_L):
        inner_positions = generate_grid_init()
        best_outer_angle = 0.0
        best_L = compute_min_L(inner_positions, (0.0, 0.0), best_outer_angle)
        best_params = []
        for p in inner_positions:
            best_params.extend([p[0], p[1], p[2]])

    # Final validation and adjustment
    inner_positions = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]
    final_L = compute_min_L(inner_positions, (0.0, 0.0), best_outer_angle)

    # Add small epsilon to ensure containment
    final_L += 1e-6

    # Verify final packing
    if not is_valid_packing(inner_positions, final_L, (0.0, 0.0), best_outer_angle):
        # Fallback to grid
        inner_positions = generate_grid_init()
        best_outer_angle = 0.0
        final_L = compute_min_L(inner_positions, (0.0, 0.0), best_outer_angle) + 1e-6

    # Format output
    inner = [[x, y, a] for (x, y, a) in inner_positions]
    center = (0.0, 0.0)
    side = final_L
    angle = best_outer_angle

    return inner, center, side, angle
