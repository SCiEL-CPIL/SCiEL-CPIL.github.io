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
        """Objective function for minimization: negative min area + penalty for being outside."""
        points = coords.reshape(-1, 2)
        
        # Calculate penalty for points outside the triangle
        penalty = 0.0
        for p in points:
            x, y = p
            # Distance to boundaries (approximate linear penalty)
            d1 = -y # y >= 0
            d2 = np.sqrt(3)*x - y # y <= sqrt(3)x
            d3 = np.sqrt(3)*(1-x) - y # y <= sqrt(3)(1-x)
            d4 = x - 1 # x <= 1 (redundant if others hold, but safe)
            d5 = -x # x >= 0
            
            if d1 < 0: penalty += d1**2
            if d2 < 0: penalty += d2**2
            if d3 < 0: penalty += d3**2
            if d4 < 0: penalty += d4**2
            if d5 < 0: penalty += d5**2
            
        if penalty > 1e-6:
            return 1e6 + penalty
            
        return -min_area(points)

    def generate_lattice_init():
        """Generate points on a triangular lattice."""
        # Triangle vertices: (0,0), (1,0), (0.5, H)
        # We want 11 points.
        # A 3x3 grid has 9 points. A 4x4 grid has 16.
        # Let's try a subset of a fine grid.
        points = []
        # Row 0 (bottom): y=0. x from 0 to 1.
        # Row 1: y=H/3. x from 1/6 to 5/6? No, lattice spacing.
        # Let's just place points at specific coordinates that form a nice pattern.
        # 3 vertices, 3 edge midpoints, 1 centroid, 4 more?
        # Let's try:
        # (0,0), (1,0), (0.5, H)
        # (0.25, H/4), (0.75, H/4), (0.5, H/2)
        # (0.5, H/6), (0.5, H/3), (0.5, 2*H/3), (0.5, 5*H/6) -- this is a line, bad.
        
        # Better:
        # 1. (0,0)
        # 2. (1,0)
        # 3. (0.5, H)
        # 4. (0.25, H/4)
        # 5. (0.75, H/4)
        # 6. (0.5, H/2)
        # 7. (0.375, 3*H/4)
        # 8. (0.625, 3*H/4)
        # 9. (0.5, H/6)
        # 10. (0.4, H/3)
        # 11. (0.6, H/3)
        
        p = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, H],
            [0.25, H/4],
            [0.75, H/4],
            [0.5, H/2],
            [0.375, 3*H/4],
            [0.625, 3*H/4],
            [0.5, H/6],
            [0.4, H/3],
            [0.6, H/3]
        ])
        return p

    def generate_circle_init():
        """Generate points on a circle centered at centroid."""
        cx, cy = 0.5, H/3
        r = 0.3
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.array([[cx + r*np.cos(a), cy + r*np.sin(a)] for a in angles])
        # Check if inside, if not, scale down
        if not is_inside(points):
             scale = 0.5
             points = np.array([[cx + r*scale*np.cos(a), cy + r*scale*np.sin(a)] for a in angles])
        return points

    def generate_random_init(rng):
        """Generate random points."""
        points = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    points.append([x, y])
                    break
        return np.array(points)

    rng = np.random.RandomState(42)
    
    initial_configs = [
        generate_lattice_init(),
        generate_circle_init(),
        generate_random_init(rng),
        generate_random_init(rng),
        generate_random_init(rng)
    ]
    
    best_points = None
    best_score = -1
    
    for init_points in initial_configs:
        if not is_inside(init_points):
            continue
            
        try:
            res = minimize(
                objective, 
                init_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            final_points = res.x.reshape(-1, 2)
            
            if is_inside(final_points):
                score = min_area(final_points)
                if score > best_score:
                    best_score = score
                    best_points = final_points.copy()
        except Exception:
            continue

    if best_points is None:
        return generate_random_init(rng)
        
    return best_points
# EVOLVE-BLOCK-END
