import numpy as np
import random
from scipy.optimize import minimize

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def min_area(pts):
    # pts is (N, 2)
    # Vectorized area calculation
    # Expand points to (N, 1, 2) and (1, N, 2)
    # pi, pj, pk
    # Area = 0.5 * | (pj-pi) x (pk-pi) |
    
    # This creates arrays of size N x N x N which is 1331 elements, very small.
    # Let's do it vectorized for speed.
    
    # pi: (N, 1, 2)
    # pj: (1, N, 2)
    # pk: (1, 1, N, 2) -> No, we need all combinations i,j,k with i<j<k.
    # Vectorizing the "i<j<k" constraint is tricky.
    # The loop version in the seed is actually quite fast for N=11.
    # 165 iterations is nothing.
    # Let's stick to the loop but optimize it slightly or keep it.
    # The seed's loop is fine.
    
    min_a = float('inf')
    for i in range(N-2):
        pi = pts[i]
        for j in range(i+1, N-1):
            pj = pts[j]
            for k in range(j+1, N):
                pk = pts[k]
                cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def barycentric_to_cartesian(u, v):
    # P = u * B + v * C
    # Since A is (0,0), P = u*(1,0) + v*(0.5, sqrt(3)/2)
    x = u + 0.5 * v
    y = np.sqrt(3)/2 * v
    return np.array([x, y])

def objective(x):
    # x is a flat array of length 2*N: [u1, v1, u2, v2, ...]
    # Calculate penalty for being outside the triangle
    # Constraints: u >= 0, v >= 0, u + v <= 1
    u = x[0::2]
    v = x[1::2]
    
    penalty = 0.0
    # Soft constraints
    # If u < 0, penalty (u)^2
    # If v < 0, penalty (v)^2
    # If u+v > 1, penalty (u+v-1)^2
    
    neg_u = np.where(u < 0, u, 0)
    neg_v = np.where(v < 0, v, 0)
    over = np.where(u + v > 1, u + v - 1, 0)
    
    penalty = np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(over**2)
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i] = barycentric_to_cartesian(u[i], v[i])
        
    area = min_area(pts)
    
    # We want to maximize area, so minimize -area
    # Add penalty
    return -area + 10.0 * penalty

def construct():
    rng = np.random.default_rng(42)
    best_score = 0
    best_points = None
    
    # Run multiple restarts
    n_restarts = 20
    
    for r in range(n_restarts):
        # Initialize u, v randomly within the triangle
        # u, v >= 0, u+v <= 1
        # Generate random points in unit square, reject if u+v > 1
        u = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            while True:
                ru = rng.random()
                rv = rng.random()
                if ru + rv <= 1:
                    u[i] = ru
                    v[i] = rv
                    break
        
        x0 = np.zeros(2*N)
        x0[0::2] = u
        x0[1::2] = v
        
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        x_opt = res.x
        u_opt = x_opt[0::2]
        v_opt = x_opt[1::2]
        
        # Project to ensure validity if penalty didn't fully constrain
        # Clip u, v to valid region
        # This is a bit messy. Let's just evaluate the final points.
        pts = np.zeros((N, 2))
        for i in range(N):
            # Ensure inside
            ui = u_opt[i]
            vi = v_opt[i]
            # Simple projection if outside
            # For now, assume penalty worked.
            # If not, we can project.
            if ui < 0: ui = 0
            if vi < 0: vi = 0
            if ui + vi > 1:
                s = 1.0 / (ui + vi)
                ui *= s
                vi *= s
            pts[i] = barycentric_to_cartesian(ui, vi)
            
        score = min_area(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    return best_points
