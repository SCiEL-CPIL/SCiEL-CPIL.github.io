import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hex_violation(point, side):
    """
    Returns the violation of the constraint that 'point' is inside a regular hexagon
    of side length 'side' (circumradius 'side') centered at origin.
    The hexagon is defined by |x| <= s, |x/2 + sqrt(3)y/2| <= s, |x/2 - sqrt(3)y/2| <= s.
    """
    x, y = point
    v1 = abs(x) - side
    v2 = abs(x / 2.0 + np.sqrt(3) * y / 2.0) - side
    v3 = abs(x / 2.0 - np.sqrt(3) * y / 2.0) - side
    return max(0.0, v1, v2, v3)

def hex_overlap_violation(p1, p2):
    """
    Returns the violation of the non-overlap constraint between two aligned unit hexagons
    centered at p1 and p2.
    Non-overlap means p1 - p2 is NOT in the interior of H(0, 2).
    H(0, 2) is defined by |x| <= 2, |x/2 + sqrt(3)y/2| <= 2, |x/2 - sqrt(3)y/2| <= 2.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    
    v1 = 2.0 - abs(dx)
    v2 = 2.0 - abs(dx / 2.0 + np.sqrt(3) * dy / 2.0)
    v3 = 2.0 - abs(dx / 2.0 - np.sqrt(3) * dy / 2.0)
    
    # If any of these is negative, the vector is outside H(0,2), so no violation.
    # If all are positive, it is inside.
    # We want to minimize the "depth" inside.
    # The penalty is the minimum of the margins? No.
    # If it's inside, we need to push it out.
    # The "distance" to the boundary is the minimum of the values (2 - val).
    # Wait. If v1=1, v2=1, v3=1, it's deep inside.
    # If v1=-1, it's outside.
    # We want a penalty that is 0 if outside, and positive if inside.
    # Penalty = max(0, min(v1, v2, v3))?
    # If v1=1, v2=0.5, v3=2. Min is 0.5. It's inside.
    # If v1=-1, min is -1. Max(0, -1) = 0. Correct.
    return max(0.0, min(v1, v2, v3))

def objective(params, S):
    """
    params: array of size 24 (12 points * 2 coords)
    S: side length of the container eroded by 1 (so container side is S+1)
    """
    points = params.reshape((N, 2))
    cost = 0.0
    
    # Container violation
    for i in range(N):
        cost += hex_violation(points[i], S)
        
    # Overlap violation
    for i in range(N):
        for j in range(i + 1, N):
            cost += hex_overlap_violation(points[i], points[j])
            
    return cost

def initial_config(S):
    """
    Generate an initial configuration of 12 points in a triangular lattice.
    """
    # Triangular lattice spacing
    # Min distance is sqrt(3) ~ 1.732.
    # Let's use spacing 1.8 to be safe.
    spacing = 1.8
    h = spacing * np.sqrt(3) / 2.0
    
    points = []
    # 3 rows of 4? Or 4 rows of 3?
    # Let's try 3x4 grid (3 rows, 4 cols)
    # Row 0: 4 points
    # Row 1: 3 points (shifted)
    # Row 2: 4 points
    # Total 11. Need 12.
    # Maybe 4x3?
    # Row 0: 3
    # Row 1: 4
    # Row 2: 3
    # Row 3: 4 -> 14. Too many.
    
    # Let's just place them in a 3x4 grid with slight offsets.
    # Or simply a 4x3 grid.
    # 4 cols, 3 rows.
    # (0,0), (s,0), (2s,0), (3s,0)
    # (s/2, h), (1.5s, h), (2.5s, h), (3.5s, h) -> 4 points
    # (0, 2h), (s, 2h), (2s, 2h), (3s, 2h)
    # Total 12.
    
    s = spacing
    h = s * np.sqrt(3) / 2.0
    
    # Center the configuration
    # x range: 0 to 3s. Center at 1.5s.
    # y range: 0 to 2h. Center at h.
    
    for r in range(3):
        for c in range(4):
            x = c * s
            y = r * h
            if r % 2 == 1:
                x += s / 2.0
            points.append([x - 1.5 * s, y - h])
            
    return np.array(points).flatten()

def solve(S, timeout=10):
    """
    Try to find a valid configuration for container side S.
    Returns (success, params)
    """
    x0 = initial_config(S)
    
    # Use Powell method, no gradient needed
    res = minimize(objective, x0, args=(S,), method='Powell', options={'maxiter': 2000, 'disp': False})
    
    if res.fun < 1e-4:
        return True, res.x
    return False, res.x

def construct():
    start_time = time.time()
    
    # Binary search for S
    # Lower bound: Area argument S >= sqrt(12) ~ 3.46
    # Upper bound: 6.0
    low = 3.4
    high = 6.0
    best_S = high
    best_params = initial_config(high)
    
    # Check if high is feasible to ensure we have a valid fallback
    success, params = solve(high)
    if success:
        best_S = high
        best_params = params
    else:
        # If even high fails, just return the best attempt
        best_params = params

    # Binary search steps
    # Limit total time to 100s
    max_time = 100.0
    
    for _ in range(15): # ~15 steps is enough for precision
        if time.time() - start_time > max_time:
            break
            
        mid = (low + high) / 2.0
        success, params = solve(mid)
        
        if success:
            high = mid
            best_S = mid
            best_params = params
        else:
            low = mid
            
    # Final result
    # Container side s = S + 1
    s = best_S + 1.0
    points = best_params.reshape((N, 2))
    
    # Format output
    inner = []
    for p in points:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, s, angle
