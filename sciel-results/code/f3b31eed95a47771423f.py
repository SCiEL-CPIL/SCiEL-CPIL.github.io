"""Packing 12 unit hexagons in minimal regular hexagon - direct optimization."""
import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)
    hex_verts = np.array([
        [1.0, 0.0], [0.5, sqrt3/2], [-0.5, sqrt3/2],
        [-1.0, 0.0], [-0.5, -sqrt3/2], [0.5, -sqrt3/2]
    ])

    # Minkowski sum of two unit regular hexagons = regular 12-gon
    # The 12-gon has vertices at: 2 hex vertices + sums of adjacent pairs
    # For two hexagons with same orientation, the Minkowski sum is a regular 12-gon
    # with inradius = 2 * (sqrt3/2) = sqrt3 and circumradius = 2
    # Actually let's compute it properly
    # Hex1 vertices: (cos(k*60), sin(k*60)) for k=0..5
    # Hex2 vertices: same
    # Minkowski sum vertices are all sums v1+v2, but the convex hull
    # For two identical regular hexagons, the Minkowski sum is a regular 12-gon
    # with vertices at angles 0, 30, 60, 90, ... 330 degrees
    # The 12-gon: for k=0..11, vertex at angle k*30 degrees
    # radius alternates between 2 (at 0,60,120,180,240,300) and 2*cos(30)=sqrt(3) (at 30,90,...)
    # Wait, let me just compute it numerically
    h1 = hex_verts
    h2 = hex_verts
    # All pairwise sums
    all_sums = np.array([h1[i] + h2[j] for i in range(6) for j in range(6)])
    # Convex hull of all_sums
    from scipy.spatial import ConvexHull
    hull = ConvexHull(all_sums)
    minkowski_verts = all_sums[hull.vertices]
    # minkowski_verts is in CCW order
    n_mink = len(minkowski_verts)
    # Edge normals of the Minkowski sum (outward)
    mink_normals = np.zeros((n_mink, 2))
    for i in range(n_mink):
        v1 = minkowski_verts[i]
        v2 = minkowski_verts[(i+1) % n_mink]
        edge = v2 - v1
        # Outward normal (right perpendicular since CCW)
        mink_normals[i] = np.array([edge[1], -edge[0]])
        mink_normals[i] /= np.linalg.norm(mink_normals[i])
    mink_edges = np.zeros((n_mink, 2))
    for i in range(n_mink):
        mink_edges[i] = minkowski_verts[(i+1) % n_mink] - minkowski_verts[i]

    def hex_overlap_distance(c1, a1, c2, a2):
        """Return the minimum translation distance to separate two hexagons.
        c1, c2: centers (2,), a1, a2: rotation angles (radians)
        Returns 0 if overlapping/touching, positive if separated.
        """
        d = c2 - c1
        # Relative rotation
        dtheta = a2 - a1
        cd = math.cos(dtheta)
        sd = math.sin(dtheta)
        R = np.array([[cd, -sd], [sd, cd]])
        # Check all 12 edges of the Minkowski sum
        # For each edge with outward normal n and support point v:
        # The hexagons overlap if for ALL edges: n . (d - R*v) < 0
        # i.e., n . d < n . R . v for all edges
        # Minimum separation = max over edges of (n . R . v - n . d)
        # If this max is > 0, they are separated by that amount
        # If <= 0, they overlap
        Rv = R.T @ minkowski_verts  # R^T applied to each vertex, shape (12,2)
        # n . R^T . v for each edge i
        supports = mink_normals @ Rv  # shape (12,)
        proj_d = mink_normals @ d  # shape (12,)
        gaps = supports - proj_d  # shape (12,)
        return float(np.max(gaps))

    def min_L_for_hexagon(cx, cy, theta):
        """Minimum L for unit hexagon at (cx,cy) with rotation theta in outer hex of side L centered at origin."""
        ca = math.cos(theta)
        sa = math.sin(theta)
        R = np.array([[ca, -sa], [sa, ca]])
        verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
        # Outer hexagon (flat-top, side L): 6 constraints
        # Normal directions at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
        normal_cos = np.cos(normal_angles)
        normal_sin = np.sin(normal_angles)
        projs = verts[0] * normal_cos + verts[1] * normal_sin
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def outer_hex_side_from_positions(positions, thetas):
        """Compute the minimum side length L of a regular hexagon (flat-top, centered at origin)
        that contains all hexagons at given positions and rotations.
        Also try rotating the outer hexagon to minimize L.
        """
        # First, compute L for outer hex at angle 0
        # Then try other outer hex angles
        best_L = float('inf')
        for outer_angle_deg in range(0, 60, 5):
            outer_angle = math.radians(outer_angle_deg)
            L_max = 0.0
            for i in range(len(positions)):
                cx, cy = positions[i]
                theta = thetas[i]
                # Relative rotation
                rel_theta = theta - outer_angle
                L_h = min_L_for_hexagon(cx, cy, rel_theta)
                # But we need to account for outer hex rotation
                # Actually min_L_for_hexagon assumes outer hex is flat-top
                # Let's just compute directly
                ca = math.cos(theta)
                sa = math.sin(theta)
                R = np.array([[ca, -sa], [sa, ca]])
                verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
                # Outer hex normals at outer_angle + 30, 90, ...
                normal_angles = np.array([outer_angle + math.radians(30 + 60 * j) for j in range(6)])
                nc = np.cos(normal_angles)
                ns = np.sin(normal_angles)
                projs = verts[0] * nc + verts[1] * ns
                L_h = np.max(projs) / (sqrt3 / 2.0)
                if L_h > L_max:
                    L_max = L_h
            if L_max < best_L:
                best_L = L_max
        return best_L

    def outer_hex_side_fine(positions, thetas):
        """Finer search over outer hex rotation."""
        best_L = float('inf')
        best_oe = 0.0
        for oe_deg in range(0, 60, 1):
            oe = math.radians(oe_deg)
            L_max = 0.0
            for i in range(len(positions)):
                cx, cy = positions[i]
                theta = thetas[i]
                ca = math.cos(theta)
                sa = math.sin(theta)
                R = np.array([[ca, -sa], [sa, ca]])
                verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
                normal_angles = np.array([oe + math.radians(30 + 60 * j) for j in range(6)])
                nc = np.cos(normal_angles)
                ns = np.sin(normal_angles)
                projs = verts[0] * nc + verts[1] * ns
                L_h = np.max(projs) / (sqrt3 / 2.0)
                if L_h > L_max:
                    L_max = L_h
            if L_max < best_L:
                best_L = L_max
                best_oe = oe
        # Refine
        for step in [0.1, 0.01]:
            oe_start = best_oe - step
            oe_end = best_oe + step
            for oe in np.arange(oe_start, oe_end + step/2, step/10):
                L_max = 0.0
                for i in range(len(positions)):
                    cx, cy = positions[i]
                    theta = thetas[i]
                    ca = math.cos(theta)
                    sa = math.sin(theta)
                    R = np.array([[ca, -sa], [sa, ca]])
                    verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
                    normal_angles = np.array([oe + math.radians(30 + 60 * j) for j in range(6)])
                    nc = np.cos(normal_angles)
                    ns = np.sin(normal_angles)
                    projs = verts[0] * nc + verts[1] * ns
                    L_h = np.max(projs) / (sqrt3 / 2.0)
                    if L_h > L_max:
                        L_max = L_h
                if L_max < best_L:
                    best_L = L_max
                    best_oe = oe
        return best_L, best_oe

    def check_valid(positions, thetas, L, outer_angle):
        """Check if all hexagons fit in outer hex of side L at outer_angle, and don't overlap."""
        # Check outer hex containment
        for i in range(len(positions)):
            cx, cy = positions[i]
            theta = thetas[i]
            ca = math.cos(theta)
            sa = math.sin(theta)
            R = np.array([[ca, -sa], [sa, ca]])
            verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
            normal_angles = np.array([outer_angle + math.radians(30 + 60 * j) for j in range(6)])
            nc = np.cos(normal_angles)
            ns = np.sin(normal_angles)
            projs = verts[0] * nc + verts[1] * ns
            if np.max(projs) > L * sqrt3 / 2.0 + 1e-9:
                return False
        # Check non-overlap
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if hex_overlap_distance(positions[i], thetas[i], positions[j], thetas[j]) < -1e-9:
                    return False
        return True

    def gauss_seidel_fix(positions, thetas, L, outer_angle, max_iter=200):
        """Gauss-Seidel relaxation to fix overlaps and out-of-bounds."""
        positions = [list(p) for p in positions]
        thetas = list(thetas)
        for it in range(max_iter):
            changed = False
            # Fix overlaps
            for i in range(N):
                for j in range(i+1, N):
                    d = np.array(positions[j]) - np.array(positions[i])
                    dist = np.linalg.norm(d)
                    if dist < 1e-12:
                        d = np.array([1.0, 0.0])
                        dist = 1.0
                    # Minimum separation
                    sep = hex_overlap_distance(np.array(positions[i]), thetas[i], np.array(positions[j]), thetas[j])
                    if sep < -1e-9:
                        # Need to push apart by -sep
                        push = (-sep) / dist
                        positions[i][0] -= push * d[0] * 0.5
                        positions[i][1] -= push * d[1] * 0.5
                        positions[j][0] += push * d[0] * 0.5
                        positions[j][1] += push * d[1] * 0.5
                        changed = True
            # Fix out-of-bounds
            for i in range(N):
                cx, cy = positions[i]
                theta = thetas[i]
                ca = math.cos(theta)
                sa = math.sin(theta)
                R = np.array([[ca, -sa], [sa, ca]])
                verts = R @ hex_verts.T + np.array([cx, cy])[:, np.newaxis]
                normal_angles = np.array([outer_angle + math.radians(30 + 60 * j) for j in range(6)])
                nc = np.cos(normal_angles)
                ns = np.sin(normal_angles)
                projs = verts[0] * nc + verts[1] * ns
                max_idx = np.argmax(projs)
                if projs[max_idx] > L * sqrt3 / 2.0 + 1e-9:
                    # Push inward along the violated normal
                    excess = projs[max_idx] - L * sqrt3 / 2.0
                    n = np.array([nc[max_idx], ns[max_idx]])
                    positions[i][0] -= excess * n[0]
                    positions[i][1] -= excess * n[1]
                    changed = True
            if not changed:
                break
        return positions, thetas

    def optimize_single(positions, thetas, L, outer_angle, max_iter=300):
        """Joint optimization: minimize L by adjusting positions and rotations."""
        positions = [list(p) for p in positions]
        thetas = list(thetas)
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            # Step 1: Fix overlaps and bounds
            positions, thetas = gauss_seidel_fix(positions, thetas, L, outer_angle, max_iter=50)
            # Step 2: Compute actual required L
            L_actual, oe = outer_hex_side_fine(positions, thetas)
            if L_actual < L - 1e-9:
                L = L_actual
                outer_angle = oe
            elif L_actual > L + 1e-6:
                # Need to expand
                L = L_actual
                outer_angle = oe
            # Step 3: Try to improve by rotating individual hexagons
            improved = False
            for i in range(N):
                best_L_i = L
                best_theta_i = thetas[i]
                for dt in [-0.1, -0.05, 0.05, 0.1]:
                    test_theta = thetas[i] + dt
                    # Check if this rotation helps
                    # Quick check: does it reduce the max projection?
                    ca = math.cos(test_theta)
                    sa = math.sin(test_theta)
                    R = np.array([[ca, -sa], [sa, ca]])
                    verts = R @ hex_verts.T + np.array(positions[i])[:, np.newaxis]
                    normal_angles = np.array([outer_angle + math.radians(30 + 60 * j) for j in range(6)])
                    nc = np.cos(normal_angles)
                    ns = np.sin(normal_angles)
                    projs = verts[0] * nc + verts[1] * ns
                    L_i = np.max(projs) / (sqrt3 / 2.0)
                    # Also check overlaps with neighbors
                    valid = True
                    for j in range(N):
                        if j == i:
                            continue
                        if hex_overlap_distance(np.array(positions[i]), test_theta, np.array(positions[j]), thetas[j]) < -1e-9:
                            valid = False
                            break
                    if valid and L_i < best_L_i - 1e-8:
                        best_L_i = L_i
                        best_theta_i = test_theta
                if best_theta_i != thetas[i]:
                    thetas[i] = best_theta_i
                    improved = True
            if not improved and abs(L_actual - L) < 1e-8:
                break
        return positions, thetas, L, outer_angle

    # Initial configurations
    def make_compact_cluster():
        """12 hexagons in a compact cluster: center + 6 ring + 5 more"""
        # Hex grid coordinates
        # Center at (0,0)
        # Ring 1: 6 hexagons at distance 2 (adjacent)
        # Ring 2: 12 hexagons at distance 4
        # We want 12 total
        # Use: 1 center + 6 ring1 + 5 from ring2
        centers = []
        # Center
        centers.append([0.0, 0.0])
        # Ring 1 (6 hexagons)
        for k in range(6):
            angle = math.radians(60 * k)
            x = 2.0 * math.cos(angle)
            y = 2.0 * math.sin(angle)
            centers.append([x, y])
        # Ring 2 (select 5)
        ring2_angles = [math.radians(60 * k) for k in range(6)]
        ring2_angles += [math.radians(60 * k + 30) for k in range(6)]
        # Sort by angle and pick 5
        ring2_centers = []
        for angle in ring2_angles:
            x = 4.0 * math.cos(angle)
            y = 4.0 * math.sin(angle)
            ring2_centers.append([x, y])
        # Actually, for hex grid, ring 2 has 12 positions
        # Let me use proper hex grid
        centers = []
        # Hex grid: axial coordinates (q, r)
        # Position: x = 1.5*q, y = sqrt3*(r + q/2)
        # Adjacent hexagons have center distance 2
        # We need 12 positions with no two adjacent (distance 2)
        # Actually we CAN have adjacent ones touching
        # Let's use a 4x3 rectangular block in hex grid
        # q from -1 to 2, r from -1 to 1, but that's 12
        for q in range(-1, 3):
            for r in range(-1, 2):
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                centers.append([x, y])
        thetas = [0.0] * 12
        return centers, thetas

    def make_ring():
        """12 hexagons in a ring around center"""
        centers = []
        for k in range(12):
            angle = math.radians(30 * k)
            # Distance to make them touch
            # Two hexagons at distance d touch when d = 2 (same orientation)
            # For a ring of 12, the chord between adjacent = 2*R*sin(pi/12)
            # Set chord = 2: R = 2/(2*sin(pi/12)) = 1/sin(15 deg)
            R = 1.0 / math.sin(math.radians(15))
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            centers.append([x, y])
        thetas = [0.0] * 12
        return centers, thetas

    def make_line():
        """12 hexagons in a line"""
        centers = []
        for k in range(12):
            x = 2.0 * (k - 5.5)
            y = 0.0
            centers.append([x, y])
        thetas = [0.0] * 12
        return centers, thetas

    def make_two_rows():
        """Two rows of 6"""
        centers = []
        for row in range(2):
            for col in range(6):
                x = 2.0 * (col - 2.5)
                y = sqrt3 * row
                centers.append([x, y])
        thetas = [0.0] * 12
        return centers, thetas

    def make_hexagonal():
        """1 center + 6 ring + 5 outer"""
        centers = []
        centers.append([0.0, 0.0])
        for k in range(6):
            angle = math.radians(60 * k)
            centers.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
        # Add 5 more at distance 4
        for k in range(5):
            angle = math.radians(60 * k + 30)
            centers.append([4.0 * math.cos(angle), 4.0 * math.sin(angle)])
        thetas = [0.0] * 12
        return centers, thetas

    best_L = float('inf')
    best_positions = None
    best_thetas = None
    best_outer_angle = 0.0

    initial_configs = [
        make_compact_cluster,
        make_ring,
        make_line,
        make_two_rows,
        make_hexagonal,
    ]

    for config_fn in initial_configs:
        if time.time() > deadline - 20.0:
            break

        positions, thetas = config_fn()

        # Compute initial L
        L, outer_angle = outer_hex_side_fine(positions, thetas)

        # Optimize
        positions, thetas, L, outer_angle = optimize_single(positions, thetas, L, outer_angle, max_iter=200)

        # Final check
        if check_valid(positions, thetas, L, outer_angle):
            if L < best_L:
                best_L = L
                best_positions = [list(p) for p in positions]
                best_thetas = list(thetas)
                best_outer_angle = outer_angle

    # Try more random restarts if time permits
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        for trial in range(20):
            if time.time() > deadline - 10.0:
                break
            # Random perturbation of the best config
            if best_positions is not None:
                positions = [list(p) + rng.normal(0, 0.1, 2) for p in best_positions]
                thetas = [t + rng.normal(0, 0.1) for t in best_thetas]
            else:
                positions, thetas = make_compact_cluster()
            L, outer_angle = outer_hex_side_fine(positions, thetas)
            positions, thetas, L, outer_angle = optimize_single(positions, thetas, L, outer_angle, max_iter=150)
            if check_valid(positions, thetas, L, outer_angle):
                if L < best_L:
                    best_L = L
                    best_positions = [list(p) for p in positions]
                    best_thetas = list(thetas)
                    best_outer_angle = outer_angle

    if best_positions is None:
        # Fallback: use the grid approach
        positions, thetas = make_compact_cluster()
        L, outer_angle = outer_hex_side_fine(positions, thetas)
        best_positions = positions
        best_thetas = thetas
        best_L = L
        best_outer_angle = outer_angle

    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        angle_deg = math.degrees(best_thetas[i]) % 60.0
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_L, math.degrees(best_outer_angle) % 60.0
