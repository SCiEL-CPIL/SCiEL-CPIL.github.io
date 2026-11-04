import numpy as np
from scipy.optimize import differential_evolution

N = 11
S3 = np.sqrt(3)

def construct():
    # Bounds for each coordinate
    # x in [0, 1]
    # y in [0, S3/2]
    bounds = [(0, 1), (0, S3/2)] * N
    
    def is_inside(pt):
        x, y = pt
        if y < -1e-9: return False
        if y > S3 * x + 1e-9: return False
        if y > -S3 * (x - 1) + 1e-9: return False
        return True

    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        
        # Check if all points are inside the triangle
        for i in range(N):
            if not is_inside(pts[i]):
                return 1e10 # Large penalty for invalid position
        
        # Calculate min area
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    area = 0.5 * abs(cross)
                    if area < min_a:
                        min_a = area
        return -min_a

    # Run Differential Evolution
    # maxiter is the number of generations
    # popsize is the population size
    # We need to be careful with time. 
    # N=11, 22 vars. 
    # Let's try popsize=15, maxiter=100. 
    # Each iteration evaluates popsize * (generations) ? No, popsize is number of individuals.
    # DE evaluates roughly popsize * maxiter * (some factor) function calls.
    # 15 * 100 * ~5 = 7500 calls. Each call is O(N^3) = 11^3/6 ~ 200 ops. Very fast.
    # We can afford more iterations.
    
    try:
        res = differential_evolution(objective, bounds, maxiter=200, popsize=20, seed=42, polish=False)
        if res.success or res.fun < 1e9:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    # Fallback: Simple random search if DE fails or times out (unlikely to timeout with these params)
    best_score = -1e10
    best_pts = None
    for _ in range(1000):
        # Rejection sampling for points inside triangle
        pts = []
        for _ in range(N):
            while True:
                x = np.random.rand()
                y = np.random.rand() * S3/2
                if is_inside([x, y]):
                    pts.append([x, y])
                    break
        pts = np.array(pts)
        # Calculate score
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    area = 0.5 * abs(cross)
                    if area < min_a:
                        min_a = area
        if min_a > -best_score:
            best_score = -min_a
            best_pts = pts
    return best_pts
