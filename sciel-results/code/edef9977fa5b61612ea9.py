import numpy as np

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Search for optimal ratio x = R2/R1
    best_x = 1.0
    best_score = 0.0
    
    def get_score(x):
        R1 = 1.0
        R2 = x
        # Angles for 20 vertices
        # Even indices: 0, 2, ..., 18 -> angles 0, 2pi/10, ...
        # Odd indices: 1, 3, ..., 19 -> angles pi/10, 3pi/10, ...
        
        # Generate vertices
        angles = np.zeros(20)
        radii = np.zeros(20)
        for i in range(20):
            if i % 2 == 0:
                angles[i] = (i // 2) * 2 * np.pi / 10
                radii[i] = R1
            else:
                angles[i] = ((i // 2) + 0.5) * 2 * np.pi / 10
                radii[i] = R2
                
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)
        points = np.stack([x_coords, y_coords], axis=1)
        
        # Calculate distances from each vertex
        min_score = 1.0
        for i in range(20):
            # Distances to all other vertices
            diffs = points - points[i]
            dists = np.linalg.norm(diffs, axis=1)
            # Remove self distance (0)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            # Find min gap
            gaps = np.diff(dists)
            min_gap = np.min(gaps)
            k = np.argmin(gaps)
            
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1:
                return 0.0
            
            # Select 4 distances
            if k == 0:
                d_sel = dists[0:4]
            elif k == 17:
                d_sel = dists[15:19]
            else:
                d_sel = dists[k-1:k+3]
                
            # Check if any d < 0.05
            if np.any(d_sel < 0.05):
                return 0.0
                
            ratios = np.maximum(d_sel / D, D / d_sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            if score < min_score:
                min_score = score
                
        return min_score

    # Coarse search
    for x in np.linspace(0.2, 3.0, 280):
        s = get_score(x)
        if s > best_score:
            best_score = s
            best_x = x
            
    # Fine search around best_x
    if best_score > 0:
        fine_xs = np.linspace(best_x - 0.05, best_x + 0.05, 100)
        for x in fine_xs:
            if x <= 0: continue
            s = get_score(x)
            if s > best_score:
                best_score = s
                best_x = x

    # Construct final polygon with best_x
    R1 = 1.0
    R2 = best_x
    angles = np.zeros(20)
    radii = np.zeros(20)
    for i in range(20):
        if i % 2 == 0:
            angles[i] = (i // 2) * 2 * np.pi / 10
            radii[i] = R1
        else:
            angles[i] = ((i // 2) + 0.5) * 2 * np.pi / 10
            radii[i] = R2
            
    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)
    points = np.stack([x_coords, y_coords], axis=1)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return points # Should not happen
    points = points / mean_dist
    
    return points
