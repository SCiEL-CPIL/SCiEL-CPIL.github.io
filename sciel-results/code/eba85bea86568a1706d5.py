import numpy as np

def construct():
    # Define 12 hexes in a 3x4 block
    # Axial coordinates
    hexes = []
    for i in range(3):
        for j in range(4):
            hexes.append((i, j))
    
    # Convert to Cartesian
    # Pointy top: x = sqrt(3)(q + r/2), y = 3r/2
    pts = []
    for q, r in hexes:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        pts.append([x, y])
    pts = np.array(pts)
    
    # Optimize outer hexagon angle theta
    # L(theta) = 1/sqrt(3) * max_k W(theta + k*60)
    # W(phi) is the width of the cluster in direction phi
    
    def get_width(pts, phi):
        u = np.array([np.cos(phi), np.sin(phi)])
        projections = pts @ u
        return np.max(projections) - np.min(projections)
    
    best_L = float('inf')
    best_theta = 0
    
    # Search theta in [0, 60)
    # Since the shape is not symmetric, we scan.
    thetas = np.linspace(0, np.pi/3, 1000)
    for theta in thetas:
        # Directions of outer hexagon sides
        dirs = [theta, theta + np.pi/3, theta + 2*np.pi/3]
        widths = [get_width(pts, d) for d in dirs]
        max_w = max(widths)
        L = max_w / np.sqrt(3)
        if L < best_L:
            best_L = L
            best_theta = theta
            
    # Construct inner hexes in world coords
    # We need to rotate the points by best_theta?
    # No, the outer hexagon is rotated by best_theta.
    # The inner hexes are fixed at angle 0.
    # The outer hexagon center is the centroid of the points?
    # To minimize L, the outer hexagon should be centered on the "center of mass" or similar?
    # Actually, for a given shape, the minimal enclosing hexagon is centered at the centroid of the shape?
    # Not necessarily. But for convex shapes, the center of the minimal enclosing hexagon is unique.
    # We can approximate by centering on the mean of the points.
    
    center = np.mean(pts, axis=0)
    
    # Return inner hexes (x, y, angle)
    # The inner hexes are at pts, with angle 0.
    inner = []
    for p in pts:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (center[0], center[1]), best_L, best_theta
