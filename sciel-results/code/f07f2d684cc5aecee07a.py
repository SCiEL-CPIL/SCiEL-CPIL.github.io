<<<<<<< SEARCH
def _params_to_points(params):
    """Convert 2*N params (barycentric u,v per point) to N Cartesian points."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        points[i] = np.array([u + 0.5 * v, v * H])
    return points
=======
def _params_to_points(params):
    """Convert 2*N params (barycentric u,v per point) to N Cartesian points, ensuring validity."""
    points = np.zeros((N, 2))
    # Define triangle vertices in Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        
        # Clamp u and v to be non-negative
        u = max(0.0, u)
        v = max(0.0, v)
        
        # If u+v > 1, project onto the line u+v=1 (edge BC)
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        
        # Convert to Cartesian
        x = u + 0.5 * v
        y = v * H
        
        # Strict boundary check and projection if still outside due to floating point or logic
        # Triangle inequalities for barycentric (u, v, w=1-u-v):
        # 1. y >= 0  => v >= 0 (handled)
        # 2. y <= sqrt(3)*x => v*H <= sqrt(3)*(u + 0.5v) => v*sqrt(3)/2 <= sqrt(3)u + sqrt(3)/2 v => 0 <= sqrt(3)u => u >= 0 (handled)
        # 3. y <= -sqrt(3)*(x-1) => v*H <= -sqrt(3)*x + sqrt(3) => v*sqrt(3)/2 <= -sqrt(3)(u + 0.5v) + sqrt(3)
        #    => v/2 <= -u - 0.5v + 1 => u + v <= 1 (handled)
        
        # However, due to floating point errors, we might still have tiny violations.
        # Let's do a final check against the actual triangle edges in Cartesian.
        # Edge AB: y >= 0
        if y < 0:
            y = 0.0
        # Edge AC: y <= sqrt(3) * x  =>  x >= y / sqrt(3)
        if x < y / np.sqrt(3):
            x = y / np.sqrt(3)
        # Edge BC: y <= -sqrt(3) * (x - 1) => x <= 1 - y / sqrt(3)
        if x > 1 - y / np.sqrt(3):
            x = 1 - y / np.sqrt(3)
            
        # If the projection resulted in an invalid state (e.g. y became negative after x clamp),
        # fallback to centroid
        if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9 or y > H + 1e-9:
             x, y = 0.5, H/3
             
        points[i] = np.array([x, y])
    return points
>>>>>>> REPLACE

