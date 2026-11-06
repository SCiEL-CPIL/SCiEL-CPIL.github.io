import numpy as np
from scipy.optimize import minimize_scalar, minimize

def construct():
    n = 8
    
    # Strategy: Optimize a "Two Squares" configuration with D4 symmetry.
    # Points are arranged as two squares, one rotated by 45 degrees.
    # Square 1 has radius R1, Square 2 has radius R2.
    # We fix R1 = 1 and optimize R2 = t.
    # The score is invariant to scaling, so we can fix R1.
    
    def get_points(t):
        # Square 1: Angles 0, pi/2, pi, 3pi/2, Radius 1
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r1 = 1.0
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2: Angles pi/4, 3pi/4, 5pi/4, 7pi/4, Radius t
        angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        r2 = t
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        points = np.vstack([p1, p2])
        return points

    def calc_score(points):
        # Calculate all pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d)
        dists = np.array(dists)
        
        D = np.max(dists)
        if D < 1e-10:
            return -np.inf
            
        # Log product - 28 * log D
        # Score = exp(sum(log(dists)) - 28*log(D))
        # We maximize log(Score) = sum(log(dists)) - 28*log(D)
        log_score = np.sum(np.log(dists)) - 28 * np.log(D)
        return log_score

    def objective(t):
        # t must be positive
        if t <= 1e-6:
            return 1e10
        points = get_points(t)
        log_s = calc_score(points)
        if not np.isfinite(log_s):
            return 1e10
        return -log_s # Minimize negative log score

    # Initial guess t=1 corresponds to regular octagon
    # We search for t in a reasonable range, e.g., [0.1, 10]
    # The optimal t might be close to 1 or significantly different.
    
    # Perform scalar minimization
    res = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded', options={'xatol': 1e-12})
    
    best_t = res.x
    best_log_score = -res.fun
    
    # Fallback: If the result is bad or invalid, use regular octagon
    if not np.isfinite(best_log_score) or best_log_score < 0: # 0 is log(1), so regular octagon
        best_t = 1.0
        best_log_score = 0.0 # Regular octagon scores 1, log(1)=0

    # Generate final points
    points = get_points(best_t)
    
    # Scale to ensure coordinates are within limits (though score is scale invariant, 
    # the evaluator scales to diameter 2, so we can return any scale)
    # The evaluator scales to diameter 2. 
    # Let's scale to diameter 2 for cleanliness, though not strictly required for the score formula 
    # if the evaluator does it. But the prompt says "scaling the set to diameter 2".
    # So returning points with diameter 2 is safe.
    
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(points[i] - points[j]))
    D = np.max(dists)
    
    if D > 0:
        points = points * (2.0 / D)
        
    # Check for distinct points
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-5:
                valid = False
    if not valid:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        points = points * 1.0 # Diameter is 2

    return points.tolist()
