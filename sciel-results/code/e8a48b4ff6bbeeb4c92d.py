"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def hexagon_support_point(direction, cx, cy, side, angle_rad):
    """Support function of a regular hexagon in a given direction (unit vector)."""
    # The support function of a regular hexagon with side s and rotation theta
    # in direction u is s * max_i (u . v_i) where v_i are unit vertex directions
    # Actually for a regular hexagon, the support function is s * |cos(angle_diff)| 
    # where angle_diff is the angle between the direction and the nearest axis
    # Let's compute it properly: the hexagon is the intersection of 6 half-planes
    # h_i(x) = u_i . x <= s * apothem where apothem = s * sqrt(3)/2
    # Actually, the support function h(u) = max_{x in H} u.x
    # For a regular hexagon with side s, centered at origin, rotated by theta:
    # The vertices are at angles theta + k*pi/3 for k=0..5, at distance s from center
    # The support function is s * max_k (u . (cos(theta+k*pi/3), sin(theta+k*pi/3)))
    # = s * max_k cos(u_angle - theta - k*pi/3)
    # This equals s * max over 6 directions of the dot product
    
    u_angle = np.arctan2(direction[1], direction[0])
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    dots = side * np.cos(u_angle - angles)
    return cx * direction[0] + cy * direction[1] + np.max(dots)

def outer_hexagon_support(direction, cx, cy, side, angle_rad):
    """Support function of the outer hexagon."""
    return hexagon_support_point(direction, cx, cy, side, angle_rad)

def min_distance_to_outer(inner_verts, outer_center, outer_side, outer_angle):
    """Compute the minimum distance from inner polygon vertices to outer hexagon boundary.
    Negative means overlap. We want all vertices inside, so we need:
    for each vertex v, support_outer(u_v) >= v.u where u_v is direction from center to v... 
    Actually, a point p is inside a convex polygon if and only if for every outward normal n_i,
    n_i . (p - c) <= apothem_i.
    For a regular hexagon with side s, the apothem is s*sqrt(3)/2.
    The 6 outward normals are at angles outer_angle + k*pi/3 for k=0..5.
    """
    apothem = outer_side * np.sqrt(3) / 2.0
    normals = np.array([[np.cos(outer_angle + k * np.pi / 3.0),
                         np.sin(outer_angle + k * np.pi / 3.0)] for k in range(6)])
    # For each inner vertex, check all 6 constraints
    # p is inside if normals[k] . (p - center) <= apothem for all k
    # distance = apothem - max_k normals[k] . (p - center)
    # We want the minimum over all vertices of this distance (margin)
    min_margin = np.inf
    for v in inner_verts:
        diff = v - np.array(outer_center)
        dots = normals @ diff
        margin = apothem - np.max(dots)
        if margin < min_margin:
            min_margin = margin
    return min_margin

def check_inner_overlap(centers, angles, sides=None):
    """Check if any two inner hexagons overlap. Returns minimum separation margin."""
    if sides is None:
        sides = [1.0] * len(centers)
    min_margin = np.inf
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            # Two regular hexagons of side s1 and s2 overlap if their distance
            # is less than the minimum distance for non-overlap
            # For two unit regular hexagons, the minimum center distance for
            # non-overlap depends on relative orientation
            # A simple sufficient condition: distance > s1 + s2 (circumradius sum)
            # But we need a tighter check. Let's use the fact that two convex polygons
            # are disjoint iff there exists a separating line.
            # For simplicity in optimization, we'll use a surrogate:
            # The minimum distance between two convex polygons can be approximated
            # by the distance between their centers minus the sum of their "radii"
            # in the direction connecting centers.
            c1, c2 = np.array(centers[i]), np.array(centers[j])
            d = c2 - c1
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                min_margin = min(min_margin, 0.0)
                continue
            u = d / dist
            # Support of hex i in direction -u, support of hex j in direction u
            h1 = hexagon_support_point(-u, centers[i][0], centers[i][1], sides[i], angles[i])
            h2 = hexagon_support_point(u, centers[j][0], centers[j][1], sides[j], angles[j])
            # The gap is: h2 - h1 - dist... wait
            # The signed distance from hex i to hex j along u is:
            # max_{x in j} u.x - max_{x in i} u.x
            # = support_j(u) - support_i(u)
            # If this is > dist, they are separated. If < dist, they overlap.
            # Actually: they are separated if support_i(-u) + support_j(u) <= dist
            # where support is relative to their centers
            s1 = hexagon_support_point(-u, 0, 0, sides[i], angles[i])
            s2 = hexagon_support_point(u, 0, 0, sides[j], angles[j])
            margin = dist - (s1 + s2)
            if margin < min_margin:
                min_margin = margin
    return min_margin

def objective(params):
    """Objective: minimize outer side length, with penalties for constraint violations."""
    # params: [outer_cx, outer_cy, outer_angle, outer_side, 
    #          x1, y1, a1, ..., x12, y12, a12]
    outer_cx, outer_cy, outer_angle, outer_side = params[0], params[1], params[2], params[3]
    
    # Inner hexagon parameters
    inner_params = params[4:]
    centers = []
    angles = []
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        centers.append([x, y])
        angles.append(a)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    # Compute all inner vertices
    inner_verts = []
    for i in range(N):
        verts = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        inner_verts.append(verts)
    inner_verts = np.vstack(inner_verts)
    
    # Penalty for inner vertices outside outer hexagon
    margin_outer = min_distance_to_outer(inner_verts, (outer_cx, outer_cy), outer_side, outer_angle)
    penalty_outer = max(0, -margin_outer) ** 2
    
    # Penalty for inner hexagon overlaps
    margin_inner = check_inner_overlap(centers, angles)
    penalty_inner = max(0, -margin_inner) ** 2
    
    # Objective: outer side + penalties
    return outer_side + 100.0 * (penalty_outer + penalty_inner)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Initial guess: place hexagons in a ring or grid pattern
    # A good initial configuration: 6 hexagons in a ring around center, 
    # or a hexagonal packing pattern
    
    # Let's start with a simple configuration: hexagons arranged in a circle
    # with center at origin, and outer hexagon also at origin
    
    # For 12 hexagons, a good arrangement might be:
    # - 1 in center, 6 in first ring, 5 more... 
    # Actually, let's try a linear arrangement first for initialization
    
    # Better: use a hexagonal lattice arrangement
    # Unit hexagons in a hexagonal lattice have centers at:
    # (i * 3/2, j * sqrt(3) + (i % 2) * sqrt(3)/2) for flat-top orientation
    
    # Let's use a compact arrangement
    # Place 12 hexagons in a roughly circular pattern
    
    # Initial positions: arrange in 2 rows of 6
    init_centers = []
    for i in range(6):
        init_centers.append([-2.25 + i * 1.5, -0.866])
        init_centers.append([-2.25 + i * 1.5, 0.866])
    
    # Convert to parameter vector
    outer_cx, outer_cy, outer_angle, outer_side = 0.0, 0.0, 0.0, 15.0
    
    params = [outer_cx, outer_cy, outer_angle, outer_side]
    for cx, cy in init_centers:
        params.extend([cx, cy, 0.0])
    
    params = np.array(params)
    
    # Define bounds
    bounds = [(-20, 20), (-20, 20), (-np.pi, np.pi), (5, 30)]
    for i in range(N):
        bounds.extend([(-15, 15), (-15, 15), (-np.pi, np.pi)])
    
    # Use differential evolution for global search, then polish
    best_result = None
    best_value = np.inf
    
    # Phase 1: Differential Evolution
    max_time_de = 60.0
    try:
        result_de = differential_evolution(
            objective, bounds,
            maxiter=100, tol=1e-4,
            popsize=15, seed=42,
            polish=True,
            update_interval=10
        )
        if result_de.success or result_de.fun < best_value:
            best_value = result_de.fun
            best_result = result_de.x
        print(f"DE result: {result_de.fun}")
    except Exception as e:
        print(f"DE failed: {e}")
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Not enough time for polish, return best
        if best_result is not None:
            return _extract_solution(best_result)
        return _default_solution()
    
    # Phase 2: L-BFGS-B polish
    if best_result is not None:
        try:
            result_lbfgs = minimize(
                objective, best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_value:
                best_value = result_lbfgs.fun
                best_result = result_lbfgs.x
            print(f"L-BFGS-B result: {result_lbfgs.fun}")
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        if best_result is not None:
            return _extract_solution(best_result)
        return _default_solution()
    
    # Phase 3: Final high-precision polish
    if best_result is not None:
        try:
            result_final = minimize(
                objective, best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_final.fun < best_value:
                best_value = result_final.fun
                best_result = result_final.x
            print(f"Final polish result: {result_final.fun}")
        except Exception as e:
            print(f"Final polish failed: {e}")
    
    if best_result is not None:
        return _extract_solution(best_result)
    return _default_solution()

def _extract_solution(params):
    """Extract solution from parameter vector."""
    outer_cx, outer_cy, outer_angle, outer_side = params[0], params[1], params[2], params[3]
    
    inner = []
    for i in range(N):
        x, y, a = params[4 + 3*i], params[4 + 3*i + 1], params[4 + 3*i + 2]
        angle_deg = np.degrees(a) % 360.0
        inner.append([float(x), float(y), float(angle_deg)])
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(np.degrees(outer_angle) % 360.0)

def _default_solution():
    """Return a default valid solution."""
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
