import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Initial positions: hexagonal packing
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    positions[0] = [0.0, 0.0]
    
    # First ring: 6 hexagons
    for i in range(6):
        angle = i * math.pi / 3
        positions[1+i] = [sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)]
    
    # Second ring: 4 hexagons
    for idx, i in enumerate([0, 1, 3, 4]):
        angle = i * math.pi / 3
        positions[7+idx] = [2 * sqrt3 * math.cos(angle), 2 * sqrt3 * math.sin(angle)]
    
    # Estimate initial outer side
    max_r = max(np.linalg.norm(p) + 1.0 for p in positions)
    initial_side = max_r / (sqrt3 / 2) * 1.1  # 10% margin
    
    def smooth_distance_to_boundary(px, py, ocx, ocy, side):
        """Smooth measure of how far a point is inside the outer hexagon (positive = inside)."""
        # Transform to local coords
        dx = px - ocx
        dy = py - ocy
        # For hexagon with angle 0, the three pairs of parallel sides have normals at 0, 60, 120 degrees
        # Distance to each side pair
        d1 = side - abs(dx)
        d2 = side * sqrt3 / 2 - abs(dx * 0.5 + dy * sqrt3 / 2)
        d3 = side * sqrt3 / 2 - abs(-dx * 0.5 + dy * sqrt3 / 2)
        return min(d1, d2, d3)
    
    def objective(params):
        pos_flat = params[:2*N]
        ocx, ocy = params[2*N], params[2*N+1]
        side = params[2*N+2]
        angs = params[2*N+3:]
        
        positions = pos_flat.reshape(N, 2)
        
        # Objective: minimize side
        obj = side
        
        # Penalty for overlaps: sum of squared violations
        penalty = 0.0
        
        # Pairwise: minimum distance should be >= sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                min_req = sqrt3
                if d < min_req:
                    penalty += (min_req - d) ** 2
        
        # Boundary: all vertices of inner hexagons must be inside outer
        for i in range(N):
            cx, cy = positions[i]
            a = angs[i]
            cos_a = math.cos(math.radians(a))
            sin_a = math.sin(math.radians(a))
            for k in range(6):
                theta = math.radians(a) + k * math.pi / 3
                vx = cx + math.cos(theta)
                vy = cy + math.sin(theta)
                margin = smooth_distance_to_boundary(vx, vy, ocx, ocy, side)
                if margin < 0:
                    penalty += margin ** 2
        
        return obj + 50.0 * penalty
    
    # Initial params
    x0 = np.zeros(3*N + 3)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    x0[2*N] = 0.0
    x0[2*N+1] = 0.0
    x0[2*N+2] = initial_side
    for i in range(N):
        x0[2*N+3+i] = angles[i]
    
    bounds = [(None, None)] * (3*N + 3)
    bounds[2*N+2] = (2.0, None)
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    best = result.x
    best_pos = best[:2*N].reshape(N, 2)
    best_ocx = best[2*N]
    best_ocy = best[2*N+1]
    best_side = best[2*N+2]
    best_angs = best[2*N+3:]
    
    inner = [[best_pos[i][0], best_pos[i][1], best_angs[i]] for i in range(N)]
    
    return inner, (best_ocx, best_ocy), best_side, 0.0
# EVOLVE-BLOCK-END
