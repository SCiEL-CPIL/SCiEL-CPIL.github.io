import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Initial positions on a hexagonal lattice
    # Axial coordinates (q, r)
    # Convert to Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # We pick 12 points that form a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, -1), (-2, 0)
    ]
    
    sqrt3 = np.sqrt(3)
    x0 = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        x0.extend([x, y])
        
    # x0 is a flat list of 24 floats (12 x, 12 y)
    
    def cost(params):
        xs = params[0::2]
        ys = params[1::2]
        
        # Calculate required side length S
        # Vertices of each unit hexagon (angle 0)
        # Offsets: (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        offsets = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        max_s = 0.0
        for i in range(N):
            cx = xs[i]
            cy = ys[i]
            # Generate 6 vertices
            verts = offsets + np.array([cx, cy])
            vx = verts[:, 0]
            vy = verts[:, 1]
            
            # Calculate S required for each vertex
            # S >= 2/sqrt3 * |y|
            # S >= |sqrt3/2 * x + 1/2 * y|
            # S >= |sqrt3/2 * x - 1/2 * y|
            
            s1 = (2.0 / sqrt3) * np.abs(vy)
            s2 = np.abs((sqrt3/2.0) * vx + 0.5 * vy)
            s3 = np.abs((sqrt3/2.0) * vx - 0.5 * vy)
            
            s_vertex = np.maximum(s1, np.maximum(s2, s3))
            max_s = max(max_s, np.max(s_vertex))
            
        # Penalty for overlap
        penalty = 0.0
        min_dist = sqrt3
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d)**2 * 1000.0
        
        return max_s + penalty

    # Optimize
    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    opt_params = res.x
    xs = opt_params[0::2]
    ys = opt_params[1::2]
    
    # Recalculate final side length
    # We can reuse the logic or just extract from cost
    # Let's recalculate to be sure
    offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    max_s = 0.0
    for i in range(N):
        cx = xs[i]
        cy = ys[i]
        verts = offsets + np.array([cx, cy])
        vx = verts[:, 0]
        vy = verts[:, 1]
        
        s1 = (2.0 / sqrt3) * np.abs(vy)
        s2 = np.abs((sqrt3/2.0) * vx + 0.5 * vy)
        s3 = np.abs((sqrt3/2.0) * vx - 0.5 * vy)
        
        s_vertex = np.maximum(s1, np.maximum(s2, s3))
        max_s = max(max_s, np.max(s_vertex))
        
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), max_s, 0.0

# EVOLVE-BLOCK-START
def construct():
    # Delegate to the optimized version defined above
    return _construct_impl()

def _construct_impl():
    import numpy as np
    from scipy.optimize import minimize

    N = 12
    sqrt3 = np.sqrt(3)
    
    # Initial positions on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, -1), (-2, 0)
    ]
    
    x0 = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        x0.extend([x, y])
        
    def cost(params):
        xs = params[0::2]
        ys = params[1::2]
        
        offsets = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        max_s = 0.0
        for i in range(N):
            cx = xs[i]
            cy = ys[i]
            verts = offsets + np.array([cx, cy])
            vx = verts[:, 0]
            vy = verts[:, 1]
            
            s1 = (2.0 / sqrt3) * np.abs(vy)
            s2 = np.abs((sqrt3/2.0) * vx + 0.5 * vy)
            s3 = np.abs((sqrt3/2.0) * vx - 0.5 * vy)
            
            s_vertex = np.maximum(s1, np.maximum(s2, s3))
            max_s = max(max_s, np.max(s_vertex))
            
        penalty = 0.0
        min_dist = sqrt3
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d)**2 * 1000.0
        
        return max_s + penalty

    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    opt_params = res.x
    xs = opt_params[0::2]
    ys = opt_params[1::2]
    
    offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    max_s = 0.0
    for i in range(N):
        cx = xs[i]
        cy = ys[i]
        verts = offsets + np.array([cx, cy])
        vx = verts[:, 0]
        vy = verts[:, 1]
        
        s1 = (2.0 / sqrt3) * np.abs(vy)
        s2 = np.abs((sqrt3/2.0) * vx + 0.5 * vy)
        s3 = np.abs((sqrt3/2.0) * vx - 0.5 * vy)
        
        s_vertex = np.maximum(s1, np.maximum(s2, s3))
        max_s = max(max_s, np.max(s_vertex))
        
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), max_s, 0.0
# EVOLVE-BLOCK-END
