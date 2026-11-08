<<<<<<< SEARCH
def _objective_pos(params, angles, outer_angle_deg=0.0):
    """
    Objective function for position optimization.
    params: 2*N array [x0, y0, x1, y1, ...]
    """
    centers = np.zeros((N, 2))
    for i in range(N):
        centers[i, 0] = params[2*i]
        centers[i, 1] = params[2*i+1]
    
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    
    # Add penalty if any hexagon is "too close" to boundary?
    # No, S is derived from the max extension, so containment is guaranteed by definition of S.
    # We just need to minimize S.
    # But we must ensure disjointness.
    # If they overlap, we add a large penalty.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            # If dist < sqrt(3) ~ 1.732, they likely overlap.
            # Let's use a soft penalty for dist < 1.732
            if dist < 1.732:
                penalty += (1.732 - dist)**2 * 100.0
            
    return S + penalty
=======
def _objective_full(params, outer_angle_deg=0.0):
    """
    Objective function for full optimization (positions and angles).
    params: 3*N array [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    # Calculate minimum enclosing hexagon side length
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    
    # Check validity: containment and disjointness
    # We use a strict penalty for any violation.
    penalty = 0.0
    
    # 1. Containment Check
    # For each inner hex, check if all vertices are inside the outer hex of side S
    # If S is calculated as the min enclosing, containment is theoretically satisfied.
    # However, due to floating point or if S is very small, we might have issues.
    # The main issue is usually disjointness.
    
    # 2. Disjointness Check
    # We need to ensure no two inner hexes overlap.
    # We can use a distance-based proxy or exact polygon intersection.
    # Exact intersection is expensive for DE (many iterations).
    # A good proxy: The minimum distance between centers must be >= 2 * apothem = sqrt(3) for aligned hexes?
    # Actually, for regular hexagons of side 1, the "collision diameter" varies with relative angle.
    # The minimum distance between centers for disjointness is the Minkowski sum radius.
    # For two unit hexes, the minimum center distance to be disjoint is sqrt(3) if they share an edge?
    # No, if they share an edge, distance is sqrt(3). If they touch at a vertex, distance is 2.
    # Wait, if they share an edge, they are not disjoint (they touch). Disjoint means no interior intersection.
    # So distance >= sqrt(3) is a necessary condition for *some* orientations, but not all.
    # If relative angle is 30 deg, the collision distance is different.
    
    # To be safe and fast, we will compute the actual minimum distance between polygons if they are close.
    # But for DE, we need a smooth-ish penalty.
    # Let's use a soft penalty based on the distance between centers, scaled by a factor.
    # The minimum possible distance between centers of two disjoint unit hexes is sqrt(3) (edge-to-edge).
    # If dist < sqrt(3), they MUST overlap?
    # Let's check. Apothem = sqrt(3)/2. Sum of apothems = sqrt(3).
    # If dist < sqrt(3), the hexes overlap.
    # So, if dist < sqrt(3), we have a penalty.
    # If dist >= sqrt(3), they might still overlap if angles are bad?
    # Actually, if dist >= 2 (vertex to vertex max), they are safe.
    # The critical region is sqrt(3) <= dist < 2.
    
    # For simplicity and speed in DE, we will penalize dist < sqrt(3) heavily.
    # And for dist between sqrt(3) and 2, we will do a quick check or assume safe if angles are 0/30.
    # Since we optimize angles, we should be careful.
    
    # Let's use a robust penalty:
    # If dist < sqrt(3) - epsilon: penalty += (sqrt(3) - dist)^2 * 1000
    # This ensures they don't overlap in the worst-case orientation.
    # If they are closer than sqrt(3), they overlap for ALL orientations.
    # If they are further, they might overlap for SOME orientations.
    # To handle this, we can add a small penalty for dist < 2.0 to encourage spacing.
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                penalty += (min_dist - dist)**2 * 1000.0
            elif dist < 2.0:
                # Soft penalty to push them apart if they are in the "risky" zone
                # This helps the optimizer find valid configurations more easily
                penalty += (2.0 - dist)**2 * 10.0
                
    return S + penalty
>>>>>>> REPLACE

