import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

def triangle_area(p1, p2, p3):
    # p1, p2, p3 are arrays of shape (N, 2)
    # Returns array of areas
    # Vectorized calculation
    # We need to handle the indices properly
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)
    
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Vertices of the unit equilateral triangle
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Precompute indices for all triplets
    # i < j < k
    indices = list(itertools.combinations(range(N), 3))
    idx_i = np.array([t[0] for t in indices])
    idx_j = np.array([t[1] for t in indices])
    idx_k = np.array([t[2] for t in indices])
    
    def objective(x):
        # x is flat array of length 2*N
        P = x.reshape(N, 2)
        
        # Extract points for all triplets
        Pi = P[idx_i]
        Pj = P[idx_j]
        Pk = P[idx_k]
        
        # Vectors
        v1 = Pj - Pi
        v2 = Pk - Pi
        
        # Cross product (2D)
        # z-component of cross product
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Penalty for points outside triangle
        # Check if point is inside triangle
        # Using barycentric coordinates or half-plane checks
        # Half-plane check:
        # For edge A-B, point C must be on the same side as the third vertex.
        # Let's use a simple projection or penalty.
        # Actually, Nelder-Mead can handle boundaries if the objective is bad outside.
        # But "bad" needs to be defined.
        # Let's add a penalty if any point is outside.
        
        # Function to check if points are inside
        # We can compute the barycentric coordinates for each point.
        # Or simply check if the point is to the left of each directed edge.
        # Edges: v0->v1, v1->v2, v2->v0
        # Normal to v0->v1 (1,0) is (0,1). Point (x,y) must have y >= 0.
        # Normal to v1->v2 (-0.5, sqrt(3)/2). 
        # Let's use the signed distance or just a penalty based on distance to boundary?
        # Simpler: If a point is outside, return a large negative value (so minimizer avoids it).
        # Or just project it? Projecting breaks differentiability but NM is derivative-free.
        # Let's just return a very small min_area (or large negative) if any point is outside.
        
        # Check containment
        # p is inside if:
        # 1. p is to the left of v0->v1
        # 2. p is to the left of v1->v2
        # 3. p is to the left of v2->v0
        
        # Cross product (B-A) x (P-A)
        
        def inside_tri(P):
            # P shape (N, 2)
            # Edge 0: v0(0,0) -> v1(1,0). Vector (1,0). 
            # Cross((1,0), (x,y)) = 1*y - 0*x = y. Must be >= 0.
            c0 = P[:, 1]
            
            # Edge 1: v1(1,0) -> v2(0.5, sqrt(3)/2). Vector (-0.5, sqrt(3)/2).
            # P - v1 = (x-1, y).
            # Cross((-0.5, sqrt(3)/2), (x-1, y)) = -0.5*y - (sqrt(3)/2)*(x-1)
            # Must be >= 0.
            c1 = -0.5 * P[:, 1] - (np.sqrt(3)/2) * (P[:, 0] - 1)
            
            # Edge 2: v2(0.5, sqrt(3)/2) -> v0(0,0). Vector (-0.5, -sqrt(3)/2).
            # P - v2 = (x-0.5, y-sqrt(3)/2).
            # Cross((-0.5, -sqrt(3)/2), (x-0.5, y-sqrt(3)/2))
            # = (-0.5)*(y-sqrt(3)/2) - (-sqrt(3)/2)*(x-0.5)
            # = -0.5*y + sqrt(3)/4 + sqrt(3)/2*x - sqrt(3)/4
            # = sqrt(3)/2*x - 0.5*y
            # Must be >= 0.
            c2 = (np.sqrt(3)/2) * P[:, 0] - 0.5 * P[:, 1]
            
            return np.all(c0 >= -1e-9) and np.all(c1 >= -1e-9) and np.all(c2 >= -1e-9)

        if not inside_tri(P):
            return 1e6  # Large positive value for minimization (bad)
            
        return -min_area

    best_x = None
    best_val = -1e9  # We want to maximize min_area, so minimize -min_area. 
                      # Wait, objective returns -min_area. 
                      # If min_area is 0.01, obj is -0.01.
                      # If min_area is 0.02, obj is -0.02.
                      # We want the smallest obj value (most negative).
                      # So we track the minimum obj value.
                      # Initial best_val should be +inf.
    
    best_obj = np.inf
    rng = np.random.RandomState(42)
    
    # Initial guesses
    # 1. Random points inside
    # 2. Grid points?
    
    num_restarts = 0
    while time.time() < deadline:
        # Generate random initial points
        # Barycentric coordinates u, v >= 0, u+v <= 1
        u = rng.rand(N)
        v = rng.rand(N)
        # Rejection sampling or mapping
        # u, v are in [0,1]. If u+v > 1, reflect? 
        # Simpler: generate u, v in [0,1]. If u+v > 1, swap? No.
        # Just resample until valid?
        # Or use: 
        # r1 = rng.rand(N), r2 = rng.rand(N)
        # If r1+r2 > 1: r1 = 1-r1, r2 = 1-r2
        # This maps uniformly to triangle?
        # Yes, reflecting the square into the triangle covers it uniformly?
        # Actually, the area of the triangle in (u,v) space is 0.5. The square is 1.
        # The mapping (u,v) -> (u,v) if u+v<=1 and (1-u, 1-v) if u+v>1
        # covers the triangle twice? No.
        # Let's just use rejection sampling for simplicity, it's fast enough for init.
        
        P_init = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            # Barycentric to Cartesian
            # P = u*v0 + v*v1 + (1-u-v)*v2
            # v0=(0,0), v1=(1,0), v2=(0.5, sqrt(3)/2)
            P_init[i] = u * v1 + (1 - u - v) * v2
            # Note: v0 is 0, so term drops.
            # x = u*1 + (1-u-v)*0.5 = u + 0.5 - 0.5u - 0.5v = 0.5 + 0.5u - 0.5v
            # y = 0 + (1-u-v)*sqrt(3)/2
            
        x0 = P_init.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        except Exception:
            continue
            
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
            
        num_restarts += 1
        # Limit restarts if time is tight
        if num_restarts > 50:
            break

    if best_x is None:
        # Fallback to random
        P_init = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            P_init[i] = u * v1 + (1 - u - v) * v2
        return P_init

    return best_x.reshape(N, 2)
