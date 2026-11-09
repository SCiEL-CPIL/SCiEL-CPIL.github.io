"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Precompute unit hexagon vertices for various angles
    def hex_vertices(angle_deg):
        """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
        a = math.radians(angle_deg)
        return [(math.cos(a + math.radians(60*k)), math.sin(a + math.radians(60*k))) for k in range(6)]

    def required_L(centers, angles, outer_angle=0.0):
        """Find minimum outer hexagon side L (flat-top, centered at origin)
        such that all unit hexagons at given centers with given angles fit inside."""
        outer_a = math.radians(outer_angle)
        # Outer hexagon normals (for flat-top): at angles 30, 90, 150, 210, 270, 330 + outer_angle
        normals = [(math.cos(outer_a + math.radians(60*k + 30)),
                    math.sin(outer_a + math.radians(60*k + 30))) for k in range(6)]

        min_L = 0.0
        for (cx, cy), ang in zip(centers, angles):
            verts = hex_vertices(ang)
            for (vx, vy) in verts:
                px, py = cx + vx, cy + vy
                for (nx, ny) in normals:
                    proj = px * nx + py * ny
                    L_needed = proj / (sqrt3 / 2.0)
                    if L_needed > min_L:
                        min_L = L_needed
        return min_L

    # Check if two unit hexagons overlap
    def hex_overlap(cx1, cy1, a1, cx2, cy2, a2):
        """Check if two unit hexagons overlap (approximate via center distance).
        Two unit hexagons with same orientation don't overlap if center distance > sqrt(3).
        With different orientations, it's more complex. Use conservative check."""
        dx, dy = cx2 - cx1, cy2 - cy1
        dist = math.sqrt(dx*dx + dy*dy)
        # Minimum distance for non-overlap of two unit hexagons (worst case rotation)
        # The maximum extent of a unit hexagon is 2 (vertex to vertex through center)
        # For safety, require dist > 2*1.0 (diameter) for very different orientations
        # But for same orientation, sqrt(3) is sufficient
        # Use a conservative threshold based on angle difference
        da = abs(a1 - a2) % 360
        if da > 180: da = 360 - da
        # When angles differ by 30 degrees, the minimum separation is larger
        # For simplicity, use sqrt(3) * 1.001 as base, add margin for rotation diff
        threshold = sqrt3 * 1.001
        return dist < threshold

    # Local optimization: adjust positions and angles to minimize L
    def local_optimize(centers, angles, max_iters=2000, lr=0.01):
        """Gradient-free local optimization using coordinate descent."""
        centers = list(centers)
        angles = list(angles)
        current_L = required_L(centers, angles)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Try small perturbations in x, y, angle for each hexagon
            for i in range(N):
                # Try angle perturbation
                for da in [15, -15, 30, -30, 45, -45, 60, -60, 10, -10, 5, -5]:
                    new_angles = angles[:]
                    new_angles[i] = (angles[i] + da) % 360
                    new_L = required_L(centers, new_angles)
                    if new_L < current_L - 1e-9:
                        # Check no overlaps introduced
                        ok = True
                        for j in range(N):
                            if i != j and hex_overlap(centers[i][0], centers[i][1], new_angles[i],
                                                      centers[j][0], centers[j][1], new_angles[j]):
                                ok = False
                                break
                        if ok:
                            angles = new_angles
                            current_L = new_L
                            improved = True
                            break

                # Try position perturbation
                for dx in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                    for dy in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                        new_centers = centers[:]
                        new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                        new_L = required_L(new_centers, angles)
                        if new_L < current_L - 1e-9:
                            ok = True
                            for j in range(N):
                                if i != j and hex_overlap(new_centers[i][0], new_centers[i][1], angles[i],
                                                          centers[j][0], centers[j][1], angles[j]):
                                    ok = False
                                    break
                            if ok:
                                centers = new_centers
                                current_L = new_L
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                # Try larger steps
                for i in range(N):
                    for dx in [0.1, -0.1, 0.2, -0.2]:
                        for dy in [0.1, -0.1, 0.2, -0.2]:
                            new_centers = centers[:]
                            new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                            new_L = required_L(new_centers, angles)
                            if new_L < current_L - 1e-9:
                                ok = True
                                for j in range(N):
                                    if i != j and hex_overlap(new_centers[i][0], new_centers[i][1], angles[i],
                                                              centers[j][0], centers[j][1], angles[j]):
                                        ok = False
                                        break
                                if ok:
                                    centers = new_centers
                                    current_L = new_L
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                break

        return centers, angles, current_L

    # Try all subsets of size 12 from 19 centers
    from itertools import combinations

    best_L = float('inf')
    best_config = None

    # First pass: find best subset with initial optimization
    count = 0
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        count += 1

        centers = []
        for i in subset:
            x, y, _, _ = all_centers[i]
            centers.append((x, y))
        angles = [0.0] * N

        # Quick check without optimization
        L_quick = required_L(centers, angles)
        if L_quick >= best_L * 1.01:
            continue  # Skip if clearly worse

        # Local optimization
        opt_centers, opt_angles, opt_L = local_optimize(centers, angles, max_iters=500)

        if opt_L < best_L:
            best_L = opt_L
            best_config = (opt_centers, opt_angles)

    if best_config is None:
        # Fallback
        centers = [(all_centers[i][0], all_centers[i][1]) for i in range(12)]
        angles = [0.0] * N
        best_L = required_L(centers, angles)
        best_config = (centers, angles)

    # Final optimization with more iterations
    centers, angles, final_L = local_optimize(best_config[0], best_config[1], max_iters=3000, lr=0.005)

    # Ensure strict non-overlap by adding tiny separation if needed
    # Scale positions slightly from centroid
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N

    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])

    # Center the configuration at origin
    cx_mean = sum(p[0] for p in inner) / N
    cy_mean = sum(p[1] for p in inner) / N
    inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]

    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
