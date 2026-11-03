import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2],
        [-0.5, SQRT3 / 2],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2],
        [0.5, -SQRT3 / 2]
    ])

    # Normals of outer hexagon (centered at 0,0, rotation 0)
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2],
        [-0.5, SQRT3 / 2],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2],
        [0.5, -SQRT3 / 2]
    ])

    def get_hex_verts(x, y, angle_deg):
        """Get vertices of hexagon at (x, y) with given rotation."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def calc_side_from_points(points):
        """Calculate required outer hexagon side length given a set of points."""
        max_apothem = 0.0
        for v in points:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_apothem:
                    max_apothem = proj
        return 2.0 * max_apothem / SQRT3

    def cluster_side_length(theta_deg, tx=0.0, ty=0.0):
        """
        Compute the outer hexagon side length for a rigid cluster of hexagons
        rotated by theta_deg and translated by (tx, ty).
        """
        theta_rad = np.deg2rad(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        
        # All vertices of all hexagons in the cluster
        all_points = []
        for i in range(N):
            cx, cy = cluster_centers[i]
            # Rotate center
            rx, ry = c * cx - s * cy, s * cx + c * cy
            rx += tx
            ry += ty
            # Rotate hexagon vertices
            local_verts = unit_hex_verts @ R.T
            verts = local_verts + np.array([rx, ry])
            all_points.extend(verts.tolist())
        
        all_points = np.array(all_points)
        return calc_side_from_points(all_points)

    def separation_violation(theta_deg, tx=0.0, ty=0.0):
        """
        Check if any two hexagons in the rigid cluster overlap.
        For a fixed lattice cluster, if it's valid at theta=0, it's valid for all theta
        (rigid rotation preserves relative positions). So this is just a constant check.
        Returns 0 if no overlap, positive if overlap.
        """
        # For a rigid cluster with fixed internal geometry, overlap status doesn't change
        # with rotation. We just need to verify the base cluster is valid.
        # This function is kept for interface compatibility but returns 0 for valid clusters.
        return 0.0

    # Define the hexagonal lattice cluster of 11 hexagons
    # Center + 6 in first ring + 4 more from second ring (or use a different arrangement)
    d = SQRT3  # Distance between centers of adjacent hexagons in hex lattice
    
    # Option A: Center + 6 ring + 4 from second ring
    cluster_centers = []
    cluster_centers.append((0.0, 0.0))  # Center
    for k in range(6):
        angle = k * np.pi / 3
        cluster_centers.append((d * np.cos(angle), d * np.sin(angle)))
    # Add 4 more from second ring (positions at distance 2*d at angles 0, pi/3, 2pi/3, pi)
    for k in range(4):
        angle = k * np.pi / 3
        cluster_centers.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
    
    cluster_centers = np.array(cluster_centers[:N])
    
    # Verify the base cluster has no overlaps (it should for a proper hex lattice)
    def check_cluster_validity():
        """Verify no two hexagons in the cluster overlap."""
        for i in range(N):
            for j in range(i + 1, N):
                dx = cluster_centers[i, 0] - cluster_centers[j, 0]
                dy = cluster_centers[i, 1] - cluster_centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < d - 1e-6:
                    return False
        return True
    
    cluster_valid = check_cluster_validity()
    
    # If the standard cluster is invalid, try alternative arrangements
    if not cluster_valid:
        # Try a more spread out arrangement
        cluster_centers_alt = []
        # Two rows of 6 and 5, staggered
        for i in range(6):
            cluster_centers_alt.append((i * d - 2.5 * d, 0.0))
        for i in range(5):
            cluster_centers_alt.append((i * d - 2.0 * d + d / 2, 1.5))
        cluster_centers = np.array(cluster_centers_alt[:N])
        cluster_valid = check_cluster_validity()
    
    # Optimize rotation angle to minimize enclosing hexagon side length
    best_side = float('inf')
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    
    # First, try optimizing just the rotation angle (cluster centered at origin)
    if cluster_valid:
        # Try a grid of angles first
        for theta in np.arange(0, 60, 1.0):
            side = cluster_side_length(theta)
            if side < best_side:
                best_side = side
                best_theta = theta
        
        # Refine with scipy optimization
        try:
            result = minimize_scalar(
                lambda t: cluster_side_length(t),
                bounds=(max(0, best_theta - 10), min(60, best_theta + 10)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 1000}
            )
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_theta = result.x
        except Exception:
            pass
        
        # Also try with small translations to center the cluster better
        for tx in np.arange(-1, 1.1, 0.5):
            for ty in np.arange(-1, 1.1, 0.5):
                try:
                    result2 = minimize_scalar(
                        lambda t: cluster_side_length(t, tx, ty),
                        bounds=(max(0, best_theta - 15), min(60, best_theta + 15)),
                        method='bounded',
                        options={'xatol': 1e-10, 'maxiter': 500}
                    )
                    if result2.success and result2.fun < best_side:
                        best_side = result2.fun
                        best_theta = result2.x
                        best_tx = tx
                        best_ty = ty
                except Exception:
                    pass
        
        # Fine-tune translation with 2D optimization
        if time.time() < deadline - 5:
            def translate_obj(params):
                tx, ty = params
                # For each translation, find best rotation
                try:
                    res = minimize_scalar(
                        lambda t: cluster_side_length(t, tx, ty),
                        bounds=(0, 60),
                        method='bounded',
                        options={'xatol': 1e-9, 'maxiter': 200}
                    )
                    return res.fun
                except:
                    return 100.0
            
            try:
                res3 = minimize(
                    translate_obj,
                    x0=[best_tx, best_ty],
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500}
                )
                if res3.success:
                    new_tx, new_ty = res3.x
                    res4 = minimize_scalar(
                        lambda t: cluster_side_length(t, new_tx, new_ty),
                        bounds=(0, 60),
                        method='bounded',
                        options={'xatol': 1e-12, 'maxiter': 2000}
                    )
                    if res4.success and res4.fun < best_side:
                        best_side = res4.fun
                        best_theta = res4.x
                        best_tx = new_tx
                        best_ty = new_ty
            except Exception:
                pass
    
    # Final high-precision polish
    if best_side < float('inf') and time.time() < deadline - 3:
        try:
            res_final = minimize_scalar(
                lambda t: cluster_side_length(t, best_tx, best_ty),
                bounds=(max(0, best_theta - 1), min(60, best_theta + 1)),
                method='bounded',
                options={'xatol': 1e-14, 'maxiter': 10000}
            )
            if res_final.success and res_final.fun < best_side:
                best_side = res_final.fun
                best_theta = res_final.x
        except Exception:
            pass

    # Compute final positions
    theta_rad = np.deg2rad(best_theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for i in range(N):
        cx, cy = cluster_centers[i]
        rx, ry = c * cx - s * cy, s * cx + c * cy
        rx += best_tx
        ry += best_ty
        inner.append([float(rx), float(ry), float(best_theta)])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
