import numpy as np

def construct():
    N = 300
    
    # 1. Initialize with a hexagonal lattice
    # We need to generate a lattice that covers the unit circle
    # Spacing s such that we get enough points
    # Area of unit circle ~ 3.14. Area of hex cell ~ s^2 * sqrt(3)/2.
    # N ~ Area / cell_area => 300 ~ 3.14 / (0.866 * s^2) => s^2 ~ 0.012 => s ~ 0.11
    # Let's try s = 0.1
    
    s = 0.1
    k = int(1.0 / s) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep points well inside
    points = points[radii < 0.9]
    
    # If we have too many, pick random N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points if lattice is too sparse (unlikely with s=0.1)
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
                
    # 2. Inflation / Repulsion Optimization
    r = 0.02 # Start radius
    max_r = 0.06
    step = 0.005
    damping = 0.9
    
    # Pre-calculate indices for efficiency? No, just vectorize.
    
    for it in range(2000):
        # Calculate pairwise distances
        # diff[i,j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist == 0] = 1e-10
        
        # Repulsion forces
        # Force is non-zero only if dist < 2*r
        # Magnitude: (2*r - dist)
        # Direction: diff / dist
        
        force = np.zeros_like(points)
        
        # Mask for overlapping pairs
        mask = dist < 2 * r
        # Exclude self-interaction
        np.fill_diagonal(mask, False)
        
        # Calculate forces for overlapping pairs
        # We need to sum forces from all j for each i
        # F_i = sum_j (2*r - dist_ij) * (p_i - p_j) / dist_ij
        
        # Vectorized calculation
        # diff is (N, N, 2)
        # dist is (N, N)
        
        # Only compute where mask is True to save time? 
        # NumPy is fast enough for 300x300.
        
        # Term: (2*r - dist) / dist
        coeff = np.zeros_like(dist)
        coeff[mask] = (2 * r - dist[mask]) / dist[mask]
        
        # F_i = sum_j coeff[i,j] * diff[i,j]
        # diff[i,j] is p_i - p_j. So force on i is in direction p_i - p_j.
        force = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force
        # If |p_i| > 1 - r, push inwards
        radii = np.linalg.norm(points, axis=1)
        boundary_mask = radii > (1 - r)
        if np.any(boundary_mask):
            # Force direction: -p_i / |p_i|
            # Magnitude: proportional to violation
            violation = radii[boundary_mask] - (1 - r)
            # Apply force
            force[boundary_mask] -= (points[boundary_mask] / radii[boundary_mask, np.newaxis]) * violation[:, np.newaxis] * 10.0 # Strong boundary force
            
        # Update positions
        points += step * force
        
        # Keep inside circle (hard constraint)
        radii = np.linalg.norm(points, axis=1)
        too_far = radii > (1 - r)
        if np.any(too_far):
            points[too_far] = points[too_far] * ((1 - r) / radii[too_far])[:, np.newaxis]
            
        # Check if we can increase r
        # Find current min distance
        min_dist = np.min(dist[mask]) if np.any(mask) else 1.0
        min_boundary = 1 - np.max(radii)
        
        # If min_dist > 2*r and min_boundary > r, we can increase r
        # Actually, we want to increase r if the system is "relaxed" enough
        # Simple heuristic: if max force is small, increase r
        max_force = np.max(np.linalg.norm(force, axis=1))
        
        if max_force < 0.01:
            r += 0.001
            if r > max_r:
                break
        else:
            # If stuck, maybe reduce step or restart? 
            # For now, just continue, hoping it settles
            pass
            
    return points
