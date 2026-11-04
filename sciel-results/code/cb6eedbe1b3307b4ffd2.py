import numpy as np
import time
import math
import scipy.optimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    d = np.sqrt(3)

    # 11-hexagonal rigid cluster based on triangular lattice
    # Lattice vectors for unit hexagons (side 1, vertex at 0 deg):
    u = np.array([1.5, d / 2])
    v = np.array([0.0, d])

    # Lattice coordinates (i, j) for the 11 closest points to origin
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    # Use a very small scale factor to ensure strict disjointness
    # On the triangular lattice, adjacent hexagons touch at edges.
    # A 0.2% scale gives ~0.002 gap which is well above floating-point noise
    # but minimal enough to not significantly increase the bounding box.
    cluster_centers *= 1.002

    # Pre-compute all 66 vertices (11 hexagons × 6 vertices) at angle=0
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
        [-0.5, np.sqrt(3) / 2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3) / 2],
        [0.5, -np.sqrt(3) / 2]
    ])

    # All vertices for each hexagon (relative to its center)
    # Shape: (N, 6, 2)
    all_base_verts = np.tile(base_verts, (N, 1, 1))

    # For each hexagon i, its vertices in cluster frame (before global rotation)
    # verts_i = cluster_centers[i] + all_base_verts[i]
    # Shape: (N, 6, 2)
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]

    # Flatten to (66, 2)
    flat_verts = cluster_verts.reshape(-1, 2)

    def compute_side_from_angles(theta_cluster, phi_outer):
        """
        Compute the minimum outer hexagon side length given:
        - theta_cluster: global rotation of the inner cluster (degrees)
        - phi_outer: rotation of the outer hexagon (degrees)
        
        Returns the side length L.
        """
        angle_rad = np.deg2rad(theta_cluster)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])

        # Rotate all inner vertices
        rotated = flat_verts @ R.T

        # Outer hexagon normals
        phi_rad = np.deg2rad(phi_outer)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        
        # Projections of all vertices onto all 6 normals
        projs = rotated @ normals.T  # (66, 6)
        
        # The constraint is: max_k(dot(p, n_k)) <= L * sqrt(3)/2
        # So L = 2 * max_k(dot(p, n_k)) / sqrt(3)
        max_proj = np.max(projs)
        L = 2.0 * max_proj / np.sqrt(3)
        
        return L

    # Phase 1: Coarse grid search over cluster rotation (0 to 60 deg)
    # We fix outer angle to 0 initially to find a good cluster orientation
    angles_coarse = np.linspace(0, 60, 601, endpoint=False)
    sides_coarse = np.array([compute_side_from_angles(a, 0.0) for a in angles_coarse])
    best_idx = np.argmin(sides_coarse)
    best_cluster_angle = angles_coarse[best_idx]
    best_side = sides_coarse[best_idx]
    best_outer_angle = 0.0

    if time.time() > deadline - 10:
        pass  # Use best found so far

    # Phase 2: Fine grid search around best cluster angle
    fine_range = 1.0  # degrees
    angles_fine = np.linspace(best_cluster_angle - fine_range, best_cluster_angle + fine_range, 2001)
    sides_fine = np.array([compute_side_from_angles(a, 0.0) for a in angles_fine])
    best_idx_fine = np.argmin(sides_fine)
    best_cluster_angle = angles_fine[best_idx_fine]
    best_side = sides_fine[best_idx_fine]

    if time.time() > deadline - 5:
        pass

    # Phase 3: Optimize outer angle phi for the best cluster angle
    # Grid search over phi (0 to 60 deg)
    phi_values = np.linspace(0, 60, 601, endpoint=False)
    best_phi_side = float('inf')
    best_phi = 0.0
    for phi in phi_values:
        s = compute_side_from_angles(best_cluster_angle, phi)
        if s < best_phi_side:
            best_phi_side = s
            best_phi = phi

    if best_phi_side < best_side:
        best_side = best_phi_side
        best_outer_angle = best_phi

    # Refine phi
    if time.time() < deadline - 3:
        phi_fine = np.linspace(best_phi - 0.5, best_phi + 0.5, 1001)
        for phi in phi_fine:
            s = compute_side_from_angles(best_cluster_angle, phi)
            if s < best_side:
                best_side = s
                best_outer_angle = phi

    # Phase 4: Joint optimization of cluster angle and outer angle using Nelder-Mead
    if time.time() < deadline - 2:
        def obj(params):
            return compute_side_from_angles(params[0], params[1])

        try:
            res = scipy.optimize.minimize(
                obj,
                np.array([best_cluster_angle, best_outer_angle]),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if 0 <= res.x[0] <= 60 and 0 <= res.x[1] <= 60:
                side_res = compute_side_from_angles(res.x[0], res.x[1])
                if side_res < best_side:
                    best_cluster_angle = float(res.x[0])
                    best_outer_angle = float(res.x[1])
                    best_side = side_res
        except Exception:
            pass

    # Phase 5: High-precision polish on both angles simultaneously
    if time.time() < deadline - 1.0:
        def polish_obj(theta):
            """theta = [cluster_angle, outer_phi], both in degrees"""
            return compute_side_from_angles(theta[0], theta[1])

        try:
            res = scipy.optimize.minimize(
                polish_obj,
                np.array([best_cluster_angle, best_outer_angle]),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if 0 <= res.x[0] <= 60 and 0 <= res.x[1] <= 60:
                side_res = compute_side_from_angles(res.x[0], res.x[1])
                if side_res < best_side:
                    best_cluster_angle = float(res.x[0])
                    best_outer_angle = float(res.x[1])
                    best_side = side_res
        except Exception:
            pass

    # Build the inner hexagon positions
    # Each inner hexagon is at cluster_centers[i] rotated by best_cluster_angle, with its own angle = best_cluster_angle
    angle_rad = np.deg2rad(best_cluster_angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])

    rotated_centers = cluster_centers @ R.T

    inner = []
    for i in range(N):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        ang = float(best_cluster_angle)
        inner.append([x, y, ang])

    # Add a tiny margin to the side length to ensure strict containment
    # due to floating point errors in the evaluator's check
    best_side += 1e-9

    return inner, (0.0, 0.0), float(best_side), float(best_outer_angle)
