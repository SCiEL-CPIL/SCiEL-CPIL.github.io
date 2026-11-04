import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Initial guess: Hexagonal lattice pattern
    # 4 rows of 3? 3 rows of 4?
    # Let's try a compact cluster.
    # 12 hexagons.
    # Pattern:
    #   . . . .
    #   . . . .
    #   . . . .
    # But shifted to fit hexagon.
    
    # Start with a grid
    positions = []
    # 3 rows of 4
    for r in range(3):
        y = (r - 1) * sqrt3
        for c in range(4):
            x = (c - 1.5) * 2.0
            if r % 2 == 1:
                x += 1.0 # Offset middle row?
            positions.append([x, y])
            
    # Flatten to array
    x0 = np.array(positions).flatten()
    
    def get_vertices(pos):
        # pos: (12, 2)
        # Returns (72, 2) vertices for all 12 hexagons
        # Unit hexagon vertices (offset 0):
        # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        verts_local = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2.0],
            [-0.5, sqrt3/2.0],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2.0],
            [0.5, -sqrt3/2.0]
        ])
        
        # Broadcast centers
        centers = pos.reshape(N, 2)
        # vertices[i, j, k] = center[i] + vert_local[j]
        # Shape (12, 6, 2)
        verts = centers[:, np.newaxis, :] + verts_local[np.newaxis, :, :]
        return verts.reshape(-1, 2)

    def objective(x_flat):
        pos = x_flat.reshape(N, 2)
        
        # 1. Calculate required container size S
        verts = get_vertices(pos)
        
        # Constraints for hexagon container (side S, centered at 0, flat-top)
        # |x| <= S
        # |y| <= S * sqrt3/2
        # |x*sqrt3/2 + y/2| <= S * sqrt3/2
        # |x*sqrt3/2 - y/2| <= S * sqrt3/2
        
        x = verts[:, 0]
        y = verts[:, 1]
        
        s1 = np.abs(x)
        s2 = np.abs(y) * (2.0 / sqrt3)
        s3 = np.abs(x * (sqrt3/2.0) + y * 0.5) * (2.0 / sqrt3)
        s4 = np.abs(x * (sqrt3/2.0) - y * 0.5) * (2.0 / sqrt3)
        
        S_req = np.max(np.concatenate([s1, s2, s3, s4]))
        
        # 2. Calculate overlap penalty
        # Distance between centers
        centers = pos
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Minimum distance for non-overlapping unit hexagons (same orientation) is 2.0
        # Penalty for d < 2.0
        # Use squared penalty for smoothness
        overlaps = np.sum(np.maximum(0, 2.0 - dists) ** 2)
        
        # Total cost
        # We want to minimize S_req, but penalize overlaps heavily
        # If there is overlap, the packing is invalid.
        # We can return S_req + large_penalty * overlaps
        # Or, if overlaps > 0, return a large value.
        # Nelder-Mead works better with continuous landscapes.
        # Let's use: Cost = S_req + 100 * overlaps
        
        return S_req + 100.0 * overlaps

    # Optimize
    # Use Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_pos = res.x.reshape(N, 2)
    
    # Recalculate S for the best position
    verts = get_vertices(best_pos)
    x = verts[:, 0]
    y = verts[:, 1]
    s1 = np.abs(x)
    s2 = np.abs(y) * (2.0 / sqrt3)
    s3 = np.abs(x * (sqrt3/2.0) + y * 0.5) * (2.0 / sqrt3)
    s4 = np.abs(x * (sqrt3/2.0) - y * 0.5) * (2.0 / sqrt3)
    S_best = np.max(np.concatenate([s1, s2, s3, s4]))
    
    # Construct inner hexagons list
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], 0.0])
        
    # Container
    center = [0.0, 0.0]
    side = float(S_best)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
