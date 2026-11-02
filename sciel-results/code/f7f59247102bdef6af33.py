<<<<<<< SEARCH
    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        centroid = np.mean(pts, axis=0)
        pts_scaled = centroid + GAP_FACTOR * (pts - centroid)
        cx, cy = np.mean(pts_scaled, axis=0)
        beta = 0.0
        D = calc_D_for_center(pts_scaled, cx, cy, beta)
        best_global_S = (D + APOTHEM) / APOTHEM
        best_global_pts = pts_scaled
        best_global_beta = beta
        best_global_center = (cx, cy)
=======
    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        pts_scaled = pts
        S, beta, center = optimize_shape(pts_scaled)
        best_global_S = S
        best_global_pts = pts_scaled
        best_global_beta = beta
        best_global_center = center
>>>>>>> REPLACE
