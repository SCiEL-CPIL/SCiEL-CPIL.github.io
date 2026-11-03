import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Unit hexagon vertices (angle 0)
    # Side length 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Initial positions for 11 hexagons
    # A compact honeycomb arrangement
    # d is distance between centers of touching hexagons
    d = np.sqrt(3)
    
    # Let's try a 3-4-3-1 or similar structure
    # Or just 1 center, 6 ring, 4 extra
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # 6 neighbors
    for i in range(6):
        a = np.deg2rad(60 * i)
        centers.append([d * np.cos(a), d * np.sin(a)])
    # 4 more, placed at distance 2d in directions 0, 120, 240? 
    # Let's place them at (2d, 0), (2d*cos(120), 2d*sin(120)), (2d*cos(240), 2d*sin(240)), and maybe one more?
    # Actually, a better initial guess is a lattice strip.
    # Let's use a lattice generation
    # Row 0: 5 hexes
    # Row 1: 4 hexes
    # Row 2: 2 hexes?
    
    # Let's use a simple grid initialization
    # 11 points in a hexagonal lattice
    # (0,0)
    # (1,0), (1,1), (1,2), (1,3) -> 5 in a row?
    
    # Let's try:
    # y=0: x = -2d, -d, 0, d, 2d (5)
    # y=1.5: x = -1.5d, -0.5d, 0.5d, 1.5d (4)
    # y=3.0: x = -d, 0, d (3) -> Total 12.
    # Remove one from the ends?
    # y=3.0: x = -0.5d, 0.5d (2) -> Total 11.
    
    centers = []
    # Row 0 (y=0)
    for i in range(5):
        x = -2*d + i*d
        centers.append([x, 0.0])
    # Row 1 (y = d * sqrt(3)/2 = 1.5)
    for i in range(4):
        x = -1.5*d + i*d
        centers.append([x, 1.5])
    # Row 2 (y = 3.0)
    for i in range(2):
        x = -0.5*d + i*d
        centers.append([x, 3.0])
        
    # Center the cluster
    mean_c = np.mean(centers, axis=0)
    centers = np.array(centers) - mean_c
    
    # Variables: x0, y0, ..., x10, y10, S
    # We fix outer center to (0,0)
    # We fix inner angles to 0
    
    def objective(vars):
        if time.time() > deadline:
            return np.inf
        xys = vars[:2*N].reshape(N, 2)
        S = vars[2*N]
        
        # 1. Overlap penalty
        # Min distance between centers must be >= d
        penalty_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = xys[i] - xys[j]
                dist = np.linalg.norm(diff)
                if dist < d:
                    penalty_overlap += (d - dist)**2
        
        # 2. Inside penalty
        # Outer hexagon: side S, center (0,0), angle 0
        # Inradius R = S * sqrt(3) / 2
        R = S * np.sqrt(3) / 2
        # Normals of outer hexagon
        # Angles 0, 60, 120, 180, 240, 300
        outer_normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        penalty_inside = 0.0
        # Vertices of each inner hexagon
        for i in range(N):
            # Vertices of inner hex i are xys[i] + unit_vertices
            verts = xys[i] + unit_vertices
            # Check against each outer normal
            # dot(verts, normals) <= R
            # We want max(0, dot - R)
            dots = verts @ outer_normals.T # Shape (6, 6)
            # For each vertex and each normal, if dot > R, penalty
            # Actually, we just need to check if the vertex is outside.
            # A point is outside if it violates ANY of the half-planes.
            # So for each vertex, find the max violation.
            # Violation = max_k (n_k . v - R)
            # Sum of squared violations
            violations = dots - R
            max_viol = np.max(violations, axis=1) # Shape (6,)
            pos_viol = np.maximum(0, max_viol)
            penalty_inside += np.sum(pos_viol**2)
            
        return S + 10.0 * (penalty_overlap + penalty_inside)

    def jac(vars):
        # Numerical gradient is slow, but L-BFGS-B can handle it if we don't provide jac?
        # Actually, providing jac is better. But let's try without first to see if it works.
        # If we don't provide jac, it uses finite differences.
        # 22 variables. 22 evaluations per step. Should be okay for a few steps.
        return None

    x0 = np.concatenate((centers.flatten(), [4.0])) # Initial S guess
    
    # Optimization
    best_S = 4.0
    best_vars = x0
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac, 
                       options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
        if res.success or res.fun < objective(x0):
            best_vars = res.x
            best_S = res.x[2*N]
            # Verify validity
            # Check if penalty is 0
            xys = best_vars[:2*N].reshape(N, 2)
            S = best_S
            # Quick check
            # If penalty is high, it's invalid
    except Exception as e:
        pass
    
    # If optimization failed or didn't improve, use initial
    # Let's do a few restarts with slight perturbations if time permits
    
    # Final result
    inner = [[xys[i,0], xys[i,1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
