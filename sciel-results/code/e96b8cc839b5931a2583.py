"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    H = np.sqrt(3) / 2
    
    def min_area(points):
        """Calculate the minimum area of any triangle formed by 3 points."""
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    a = 0.5 * abs(det)
                    if a < min_a:
                        min_a = a
        return min_a

    def is_inside(points):
        """Check if all points are inside the equilateral triangle."""
        for p in points:
            x, y = p
            if y < -1e-9: return False
            if y > np.sqrt(3)*x + 1e-9: return False
            if y > np.sqrt(3)*(1-x) + 1e-9: return False
            if x < -1e-9 or x > 1 + 1e-9: return False
        return True

    def objective(coords):
        points = coords.reshape(-1, 2)
        if not is_inside(points):
            return 1e6
        return -min_area(points)

    def random_point_in_triangle(rng):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                return np.array([u + 0.5 * v, H * v])

    def random_points(rng):
        return np.array([random_point_in_triangle(rng) for _ in range(N)])

    # Generate diverse seeds
    seeds = []
    
    # Seed 1: Symmetric structured guess
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, H])
    m1 = (v1 + v2) / 2
    m2 = (v2 + v3) / 2
    m3 = (v3 + v1) / 2
    c = (v1 + v2 + v3) / 3
    
    # 3 vertices, 3 midpoints, 1 centroid, 4 points at 0.4 distance from centroid to vertices
    pts = [v1, v2, v3, m1, m2, m3, c]
    for v in [v1, v2, v3]:
        pts.append(c + 0.4 * (v - c))
    pts.append(c + 0.2 * (v1 - c)) # 11th point
    seeds.append(np.array(pts))
    
    # Seed 2: Triangular grid approximation
    # 4x4 grid (10 pts) + 1 center
    grid_pts = []
    for i in range(4):
        for j in range(4-i):
            # Barycentric to Cartesian
            # (x, y) = i/3 * v1 + j/3 * v2 + (3-i-j)/3 * v3 ? No.
            # Let's use linear interpolation.
            # Row k (0 to 3), point j (0 to 3-k)
            pass # Skipping complex grid generation for brevity, relying on random + structured
    
    # Add 10 random seeds
    rng = np.random.RandomState(42)
    for _ in range(10):
        seeds.append(random_points(rng))

    best_points = None
    best_score = -1
    
    for seed in seeds:
        if not is_inside(seed):
            continue
            
        try:
            res = minimize(
                objective, 
                seed.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            final_points = res.x.reshape(-1, 2)
            if is_inside(final_points):
                final_score = min_area(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        return random_points(rng)
        
    return best_points
# EVOLVE-BLOCK-END
