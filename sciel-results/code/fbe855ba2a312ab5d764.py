import numpy as np

def hexagon_packing_11():
    # Generate 19-hexagon cluster (hexagon of side 3)
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= 2:
                centers.append((q, r))
    
    # Convert to Cartesian
    # Unit hexagon side 1.
    # Distance between centers is sqrt(3).
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    cart_centers = []
    for q, r in centers:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        cart_centers.append([x, y])
    cart_centers = np.array(cart_centers)
    
    # Remove 8 farthest from centroid to get 11
    centroid = np.mean(cart_centers, axis=0)
    dists = np.linalg.norm(cart_centers - centroid, axis=1)
    # Indices of farthest
    farthest_idx = np.argsort(dists)[::-1][:8]
    # Keep the rest
    keep_mask = np.ones(len(cart_centers), dtype=bool)
    keep_mask[farthest_idx] = False
    inner_centers = cart_centers[keep_mask]
    
    # Recalculate centroid of the 11
    centroid_11 = np.mean(inner_centers, axis=0)
    inner_centers -= centroid_11
    
    # Optimize theta
    best_S = float('inf')
    best_theta = 0
    # theta is the angle of the outer hexagon
    # Normals are at theta + 60k
    for theta in np.linspace(0, np.pi/3, 1000):
        S = 0
        for k in range(6):
            angle = theta + k * np.pi / 3
            n = np.array([np.cos(angle), np.sin(angle)])
            projs = inner_centers @ n
            L = np.max(projs)
            R = np.min(projs)
            W = L - R
            S_k = W / np.sqrt(3)
            if S_k > S:
                S = S_k
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Construct output
    inner_data = np.zeros((11, 3))
    for i in range(11):
        inner_data[i, 0] = inner_centers[i, 0]
        inner_data[i, 1] = inner_centers[i, 1]
        inner_data[i, 2] = 0.0 # angle in degrees
        
    outer_data = np.array([0.0, 0.0, np.degrees(best_theta)])
    
    return inner_data, outer_data, best_S
