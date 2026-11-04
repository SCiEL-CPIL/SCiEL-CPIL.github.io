import numpy as np
from scipy.optimize import minimize
import math

N = 12
DIST = math.sqrt(3)

def construct():
    # Define the 3 normal directions for the hexagon sides
    # Hexagon sides are at angles 0, 60, 120 degrees?
    # Let's use normals at 0, 60, 120.
    # Normal vectors:
    n1 = np.array([1.0, 0.0])
    n2 = np.array([0.5, math.sqrt(3)/2])
    n3 = np.array([-0.5, math.sqrt(3)/2])
    
    # Initial guess: 12 points from a hex lattice
    # Generate points in a hex lattice
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1 (6 points)
    for k in range(6):
        angle = k * math.pi / 3
        x = DIST * math.cos(angle)
        y = DIST * math.sin(angle)
        pts.append([x, y])
    # Ring 2 (12 points), take 5
    # We need 5 more to make 12.
    # Let's take the first 5 from ring 2
    for k in range(5):
        # Angle offset to avoid symmetry issues?
        angle = k * math.pi / 3 + math.pi/6 # Mid-edges of ring 1?
        # Actually, let's just use lattice points.
        # Lattice points at distance 2*DIST?
        # No, lattice spacing is DIST.
        # Ring 2 points are at distance 2*DIST from center.
        # Let's pick 5 specific ones.
        # (2,0), (1,1), (0,2), (-1,1), (-2,0) in axial coords?
        # Let's just use angles.
        angle = k * math.pi / 3
        x = 2 * DIST * math.cos(angle)
        y = 2 * DIST * math.sin(angle)
        pts.append([x, y])
        
    pts = np.array(pts)
    
    # We want to minimize S.
    # S = 1 + L
    # L = (2/sqrt(3)) * max_proj
    # max_proj = max_k max_i |(p_i - c) dot n_k|
    # We can fix c = 0 by centering the points in the objective function.
    
    def objective(v):
        P = v.reshape(N, 2)
        # Center the points
        c = np.mean(P, axis=0)
        P_centered = P - c
        
        # Calculate max projection
        max_proj = 0
        for n in [n1, n2, n3]:
            projs = np.abs(P_centered @ n)
            max_proj = max(max_proj, np.max(projs))
        
        L = (2.0 / math.sqrt(3)) * max_proj
        S = 1.0 + L
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(P[i] - P[j])
                if d < DIST:
                    penalty += (DIST - d)**2
        return S + 100.0 * penalty

    v0 = pts.flatten()
    
    # Optimize
    res = minimize(objective, v0, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    P_opt = res.x.reshape(N, 2)
    c_opt = np.mean(P_opt, axis=0)
    P_opt = P_opt - c_opt
    
    # Calculate final S
    max_proj = 0
    for n in [n1, n2, n3]:
        projs = np.abs(P_opt @ n)
        max_proj = max(max_proj, np.max(projs))
    
    L_opt = (2.0 / math.sqrt(3)) * max_proj
    S_opt = 1.0 + L_opt
    
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = P_opt[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S_opt, 0.0

