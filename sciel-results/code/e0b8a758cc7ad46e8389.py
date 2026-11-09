import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses vectorized objective calculation and scipy.optimize.minimize (Powell method).
    """
    
    def vectorized_min_triangle_area(pts):
        """
        Compute the minimum area among all triangles formed by 3 points.
        pts: shape (N, 2)
        """
        N = pts.shape[0]
        # We need to compute area for all (i, j, k) with i < j < k
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Using cross product in 2D: area = 0.5 * |det([p2-p1, p3-p1])|
        
        # This loop is O(N^3). For N=13, N^3 is 2197, which is small.
        # However, vectorizing it can be tricky with unique combinations.
        # Given N is small, a simple loop in Python might be faster than complex vectorization overhead
        # if the vectorization involves creating large intermediate arrays.
        # Let's stick to a semi-vectorized approach or just efficient loops if scipy is slow.
        # Actually, for N=13, even pure Python loops for the objective are fast enough 
        # if the optimizer doesn't call it millions of times. 
        # Powell method might call it ~1000-5000 times. 
        # 5000 * (13*12*11/6) ~ 5000 * 286 ~ 1.4M area calculations.
        # Each calculation is cheap. This should be fine.
        
        min_area = np.inf
        # To speed up, we can precompute indices
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(params):
        """
        Objective function to minimize: - (A_min / A_hull)
        params: shape (26,)
        """
        pts = params.reshape(13, 2)
        
        # Check for invalid points (NaN, Inf)
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        # Compute hull area
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull <= 1e-10:
                return 1e10
        except Exception:
            return 1e10
            
        # Compute min triangle area
        a_min = vectorized_min_triangle_area(pts)
        
        if a_min <= 1e-10:
            return 1e10
            
        ratio = a_min / a_hull
        return -ratio

    # Initial guess: Regular hexagonal lattice
    # 1 center, 6 inner, 6 outer
    s = 1.0
    sqrt3 = np.sqrt(3)
    
    # Center
    p0 = np.array([0.0, 0.0])
    
    # Inner ring (radius 1)
    inner = []
    for k in range(6):
        angle = np.pi / 3 * k
        inner.append([np.cos(angle), np.sin(angle)])
        
    # Outer ring (radius 2, offset angles)
    # For a perfect hexagonal packing, the outer points are at distance 2 from center?
    # No, in a hex grid with spacing 1:
    # Center (0,0)
    # Neighbors at dist 1: (1,0), (1/2, sqrt3/2), etc.
    # Next neighbors at dist 2? Or dist sqrt(3)?
    # Let's use the standard hexagonal numbers layout.
    # 13 points:
    # Layer 0: 1 point
    # Layer 1: 6 points
    # Layer 2: 6 points? No, Layer 2 has 12 points.
    # So 1 + 6 + 6 is not a standard hexagonal layer completion.
    # A common 13-point configuration is a hexagon of side length 2 (which has 19 points) minus the 6 corners?
    # Or just a random regular polygon?
    # Let's stick to the "1 center, 6 inner, 6 outer" structure but optimized.
    # The "6 outer" in the previous code were at 1.5s.
    # Let's initialize with a regular 12-gon + center?
    # Or better: 1 center, 6 at r=1, 6 at r=2 with alternating angles?
    
    init_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
        # Outer 6: let's place them at radius 2, angles 0, 60, ...
        # But this might collide with inner points if not careful.
        # Actually, a better init is a regular 13-gon? No, that's bad for triangles.
        # Let's use the hexagonal packing init from before, it's a good local minimum.
        [2.0, 0.0],
        [1.0, np.sqrt(3)],
        [-1.0, np.sqrt(3)],
        [-2.0, 0.0],
        [-1.0, -np.sqrt(3)],
        [1.0, -np.sqrt(3)]
    ])
    
    # Flatten for optimizer
    x0 = init_pts.flatten()
    
    # Run Powell optimization
    # Powell is robust for this type of problem.
    res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
    
    best_pts = res.x.reshape(13, 2)
    
    # Sometimes Powell can get stuck or drift. Let's try a second pass with Nelder-Mead or just return this.
    # Given the time limit, one good Powell run should be sufficient.
    
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        return init_pts
        
    return best_pts
