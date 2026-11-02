"""Packing 12 unit hexagons in minimal regular hexagon via support-function QCQP."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def hexagon_support(direction_angle, hex_angle_deg, side=1.0):
        """Support function of a regular hexagon with given side and rotation.
        direction_angle: angle of the outward normal direction (radians)
        hex_angle_deg: rotation of the hexagon (degrees)
        Returns the maximum projection of the hexagon onto that direction.
        """
        hex_angle_rad = np.radians(hex_angle_deg)
        # The hexagon has 6 face normals at angles hex_angle + 30 + 60k (degrees)
        # In the local frame, the support is: for a regular hexagon with side s,
        # the apothem is s*sqrt(3)/2, and the support in direction phi (relative to hex) is:
        # s * cos(phi - 30 - 60k) for the k-th face where |phi - 30 - 60k| <= 30
        # Equivalently: s / cos(30) * cos(min_k |phi - (30+60k)|) but capped at apothem
        
        # Simpler: the support function of a regular hexagon (side 1) with one face normal at angle 0:
        # h(phi) = cos(phi) / cos(pi/6) for |phi| <= pi/6
        #         = sqrt(3)/2 for pi/6 <= |phi| <= pi/2  (the apothem, flat region)
        # But actually for a hexagon, the support is piecewise linear in the normal directions.
        
        # For a regular hexagon with side s, the support function in direction making angle alpha
        # with the nearest face normal is: s * cos(alpha) / cos(pi/6) for |alpha| <= pi/6
        # and s * sqrt(3)/2 for pi/6 < |alpha| <= pi/2
        
        # Actually, let me think more carefully. A regular hexagon with side 1 can be seen as the
        # Minkowski sum of 3 line segments (zonotope). The vertices are at angles 0,60,120,180,240,300
        # at distance 1 from center (for a hexagon with "radius" = side length = 1).
        # Wait - for a regular hexagon with side length 1, the circumradius is 1 and inradius (apothem) is sqrt(3)/2.
        
        # The support function h(n) = max_{v in H} v·n
        # For a regular hexagon with circumradius R=1 (side=1), vertices at angles 0,60,...,300:
        # h(phi) = max_k cos(phi - k*pi/3) for k=0..5  (times R=1)
        # This equals cos(phi mod pi/3 - pi/6) / cos(pi/6) ... no.
        # h(phi) = max(cos(phi), cos(phi-pi/3), cos(phi-2pi/3), ...)
        # = cos(min_k |phi - k*pi/3|)  ... no, that's not right either.
        
        # Let me just compute it directly:
        # The hexagon vertices (side=1, circumradius=1) at angles 0,60,120,180,240,300
        # h(phi) = max_i cos(phi - i*pi/3)  for i=0..5
        # This is periodic with period pi/3, and in [0, pi/3]:
        # h(phi) = max(cos(phi), cos(phi-pi/3)) = cos(min(phi, pi/3-phi))
        # At phi=0: h=1, at phi=pi/6: h=cos(pi/6)=sqrt(3)/2
        
        # So for a hexagon rotated by theta, the support in direction phi is:
        # h(phi - theta) where h is the above function with period pi/3
        
        rel = direction_angle - hex_angle_rad
        rel = rel % (np.pi / 3)  # reduce to [0, pi/3)
        rel = min(rel, np.pi / 3 - rel)  # now in [0, pi/6]
        return side * np.cos(rel)

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_separation_axes(angle1_deg, angle2_deg):
        """Get the 12 normal directions for SAT between two hexagons with given rotations."""
        axes = []
        for a_deg in [angle1_deg, angle2_deg]:
            a_rad = np.radians(a_deg)
            for k in range(6):
                # Face normals are at angles a + 30 + 60k degrees
                na = a_rad + np.radians(30 + 60 * k)
                axes.append(na)
        return axes

    def check_disjoint(c1, a1, c2, a2, side1=1.0, side2=1.0):
        """Check if two hexagons are disjoint using support function.
        c1, c2: center positions (x,y)
        a1, a2: rotation angles in degrees
        Returns True if disjoint (non-overlapping).
        """
        d = np.array([c2[0] - c1[0], c2[1] - c1[1]])
        axes = get_separation_axes(a1, a2)
        for na in axes:
            n = np.array([np.cos(na), np.sin(na)])
            proj = np.dot(d, n)
            h1 = hexagon_support(na, a1, side1)
            h2 = hexagon_support(na, a2, side2)
            if proj > h1 + h2 + 1e-9:
                return True
        return False

    def check_disjoint_all(positions):
        """Check all pairs are disjoint. positions: list of (x, y, angle_deg)."""
        for i in range(N):
            for j in range(i + 1, N):
                if not check_disjoint(positions[i][:2], positions[i][2], positions[j][:2], positions[j][2]):
                    return False
        return True

    def compute_outer_hexagon(positions, outer_angle_deg):
        """Compute minimum outer hexagon (center, side) for given outer angle containing all inner hexagons.
        Returns (center_x, center_y, side) or None if infeasible.
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for cx, cy, ang in positions:
            verts = hexagon_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # For outer hexagon with angle oa, side s, center (ox, oy):
        # For each of 6 face normals at angles oa + 30 + 60k:
        #   max_i dot(p_i - c, n_k) <= s * sqrt(3)/2
        # This is: dot(p_i, n_k) - dot(c, n_k) <= s * sqrt(3)/2 for all i, k
        # => s >= (max_i dot(p_i, n_k) - dot(c, n_k)) / (sqrt(3)/2) for all k
        # => s >= max_k [max_i dot(p_i, n_k) - dot(c, n_k)] / (sqrt(3)/2)
        
        # To minimize s, we need to maximize min_k [dot(c, n_k) - (max_i dot(p_i, n_k))] / (sqrt(3)/2)
        # Equivalently, minimize s = max_k [max_i dot(p_i, n_k) - dot(c, n_k)] / (sqrt(3)/2)
        
        oa_rad = np.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            na = oa_rad + np.radians(30 + 60 * k)
            normals.append(np.array([np.cos(na), np.sin(na)]))
        
        # Precompute max_i dot(p_i, n_k) for each normal
        max_projs = []
        for n in normals:
            projs = all_verts @ n
            max_projs.append(np.max(projs))
        max_projs = np.array(max_projs)
        
        # We need to find c = (ox, oy) that minimizes:
        #   max_k [max_projs[k] - dot(c, n_k)]
        # This is a convex problem in c. Let's use a simple approach:
        # The optimal c is where the 6 constraints are balanced.
        
        # Use scipy to solve: minimize t subject to dot(c, n_k) - max_projs[k] >= -t for all k
        # This is a linear program! Variables: ox, oy, t
        # Minimize t
        # Subject to: -dot(n_k, c) + t <= -max_projs[k] for k=0..5
        # i.e., [-n_kx, -n_ky, 1] @ [ox, oy, t] <= -max_projs[k]
        
        from scipy.optimize import linprog
        
        A = np.zeros((6, 3))
        b = np.zeros(6)
        for k in range(6):
            A[k, 0] = -normals[k][0]
            A[k, 1] = -normals[k][1]
            A[k, 2] = 1
            b[k] = -max_projs[k]
        
        c_obj = np.array([0, 0, 1])  # minimize t
        
        res = linprog(c_obj, A_ub=A, b_ub=b, method='highs')
        
        if res.success:
            ox, oy, t = res.x
            side = t / (np.sqrt(3) / 2)
            return (ox, oy, side)
        else:
            # Fallback: use centroid
            centroid = np.mean(all_verts, axis=0)
            max_proj = 0
            for k in range(6):
                n = normals[k]
                projs = (all_verts - centroid) @ n
                max_proj = max(max_proj, np.max(projs))
            side = max_proj / (np.sqrt(3) / 2)
            return (centroid[0], centroid[1], side)

    def objective_and_constraints(params, outer_angle_deg):
        """Compute objective (outer side) and check constraints.
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
        """
        positions = []
        for i in range(N):
            positions.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        # Check disjointness
        disjoint_ok = True
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.array([positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]])
                axes = get_separation_axes(positions[i][2], positions[j][2])
                min_sep = float('inf')
                for na in axes:
                    n = np.array([np.cos(na), np.sin(na)])
                    proj = np.dot(d, n)
                    h1 = hexagon_support(na, positions[i][2], 1.0)
                    h2 = hexagon_support(na, positions[j][2], 1.0)
                    sep = proj - (h1 + h2)
                    min_sep = min(min_sep, sep)
                if min_sep < -1e-6:
                    disjoint_ok = False
                    overlap_penalty += (-min_sep) ** 2 * 100.0
        
        # Compute outer hexagon
        outer = compute_outer_hexagon(positions, outer_angle_deg)
        if outer is None:
            return 1e6, False
        
        outer_side = outer[2]
        total = outer_side + overlap_penalty
        return total, disjoint_ok

    def smooth_objective(params, outer_angle_deg):
        """Smooth objective for optimization."""
        val, _ = objective_and_constraints(params, outer_angle_deg)
        return val

    # Initial configuration: honeycomb lattice
    # For 12 hexagons in a compact arrangement:
    # Row structure: 4-4-4 with hexagonal offset
    # Spacing: for two adjacent hexagons (side 1, same angle), min center distance = 2 * apothem = sqrt(3)
    # For hexagonal packing with 60-degree angles, the lattice vectors are:
    # v1 = (sqrt(3), 0) * 2 = (2*sqrt(3), 0) for same-row neighbors
    # Actually, for two hexagons with same orientation, touching, center distance = 2*apothem = sqrt(3)
    # Wait: two unit hexagons (side 1) touching along a flat side: center distance = 2 * (sqrt(3)/2) = sqrt(3)
    # For hexagonal lattice: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    # But we need them to not overlap, so minimum distance is sqrt(3).
    
    # Let me use a proper hexagonal lattice:
    # For hexagons with angle 0 (flat top), the lattice vectors for dense packing are:
    # a1 = (2*sqrt(3), 0) -- no, that's too far
    # Actually, two hexagons with the same orientation, side 1:
    # The "width" in the x-direction (flat side to flat side) = 2 * sqrt(3)/2 = sqrt(3)
    # So center-to-center for horizontal neighbors = sqrt(3)
    # For 60-degree offset neighbors: the distance is also sqrt(3)
    
    # Hexagonal lattice:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # 12 hexagons in a hexagonal arrangement:
    # Use positions in the lattice:
    # Center at origin, then 6 at distance 1, then 6 at distance 2 (partial)
    # Or a 4x3 rectangular patch of hexagonal lattice
    
    # Let's use a 4-wide, 3-tall hexagonal grid:
    # Row 0 (y=0): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Row 1 (y=1.5): x = -sqrt3, 0, sqrt3, 2*sqrt3  (offset by sqrt3/2)
    # Row 2 (y=3.0): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    
    init_positions = []
    for row in range(3):
        y = row * 1.5
        for col in range(4):
            x = (col - 1.5) * sqrt3 + (row % 2) * (sqrt3 / 2)
            init_positions.append([x, y, 0.0])
    
    # Center the configuration
    cx_mean = np.mean([p[0] for p in init_positions])
    cy_mean = np.mean([p[1] for p in init_positions])
    for p in init_positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    # Create params vector
    init_params = []
    for p in init_positions:
        init_params.extend([p[0], p[1], p[2]])
    init_params = np.array(init_params, dtype=float)
    
    best_score = float('inf')
    best_params = init_params.copy()
    best_outer = None
    
    # Evaluate initial
    positions_init = [(init_params[3*i], init_params[3*i+1], init_params[3*i+2]) for i in range(N)]
    for oa in [0, 30, 60, 90, 120, 150]:
        outer = compute_outer_hexagon(positions_init, oa)
        if outer is not None and outer[2] < best_score:
            best_score = outer[2]
            best_params = init_params.copy()
            best_outer = (oa, outer)
    
    # Multi-start optimization with different outer angles and initial perturbations
    random.seed(42)
    np.random.seed(42)
    
    outer_angles_to_try = [0, 30, 60, 90, 120, 150]
    
    for trial in range(30):  # Multiple restarts
        if time.time() > deadline - 5:
            break
        
        oa = outer_angles_to_try[trial % len(outer_angles_to_try)]
        
        # Create perturbed initial
        if trial == 0:
            params = init_params.copy()
        else:
            params = init_params.copy()
            # Add small random perturbation
            noise_scale = 0.1 + 0.5 * (trial / 30)
            params[:3*N:3] += np.random.randn(N) * noise_scale
            params[1:3*N:3] += np.random.randn(N) * noise_scale
            # Small angle perturbation
            params[2:3*N:3] += np.random.randn(N) * 5.0  # degrees
        
        # Optimize using Nelder-Mead
        try:
            def obj(p):
                return smooth_objective(p, oa)
            
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.001, 'adaptive': True})
            
            if res.fun < best_score:
                positions = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                # Verify disjointness
                if check_disjoint_all(positions):
                    outer = compute_outer_hexagon(positions, oa)
                    if outer is not None and outer[2] < best_score:
                        best_score = outer[2]
                        best_params = res.x.copy()
                        best_outer = (oa, outer)
        except Exception:
            pass
    
    # Final refinement with more iterations on the best solution
    if best_outer is not None and time.time() < deadline - 10:
        oa_best = best_outer[0]
        try:
            def obj(p):
                return smooth_objective(p, oa_best)
            
            res = minimize(obj, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 0.001, 'fatol': 0.0001, 'adaptive': True})
            
            if res.fun < best_score:
                positions = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
                if check_disjoint_all(positions):
                    outer = compute_outer_hexagon(positions, oa_best)
                    if outer is not None and outer[2] < best_score:
                        best_score = outer[2]
                        best_params = res.x.copy()
                        best_outer = (oa_best, outer)
        except Exception:
            pass
    
    # Try additional outer angles with the best positions
    positions_best = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]
    for oa in outer_angles_to_try:
        if oa == (best_outer[0] if best_outer else -1):
            continue
        outer = compute_outer_hexagon(positions_best, oa)
        if outer is not None and outer[2] < best_score:
            best_score = outer[2]
            best_outer = (oa, outer)
    
    # Extract final result
    if best_outer is None:
        # Fallback
        oa = 0
        outer = compute_outer_hexagon(positions_best, oa)
        best_outer = (oa, outer)
        best_score = outer[2]
    
    oa_final = best_outer[0]
    oc_final = (best_outer[1][0], best_outer[1][1])
    os_final = best_outer[1][2]
    
    # Ensure validity: expand if needed
    positions_final = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]
    
    # Verify containment
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * np.sqrt(3) / 2
        for k in range(6):
            na = np.radians(30 + 60 * k)
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-6:
                return False
        return True
    
    valid = True
    for cx, cy, ang in positions_final:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, oc_final[0], oc_final[1], os_final, oa_final):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Expand
        for _ in range(100):
            os_final += 0.01
            valid = True
            for cx, cy, ang in positions_final:
                verts = hexagon_vertices(cx, cy, 1.0, ang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, oc_final[0], oc_final[1], os_final, oa_final):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
    
    # Verify disjointness
    if not check_disjoint_all(positions_final):
        # Use initial configuration as fallback
        positions_final = [(p[0], p[1], p[2]) for p in init_positions]
        for oa in outer_angles_to_try:
            outer = compute_outer_hexagon(positions_final, oa)
            if outer is not None:
                oc_final = (outer[0], outer[1])
                os_final = outer[2]
                oa_final = oa
                break
    
    inner = [[p[0], p[1], p[2]] for p in positions_final]
    
    return inner, [oc_final[0], oc_final[1]], os_final, oa_final
# EVOLVE-BLOCK-END
