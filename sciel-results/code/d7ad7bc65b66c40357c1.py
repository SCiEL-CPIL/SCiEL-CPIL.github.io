import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def construct():
    # Spectral/Fourier approach
    # We parameterize the radius r(theta) = 1 + sum_{k=1}^K (a_k cos(k*theta) + b_k sin(k*theta))
    # This ensures the points are star-shaped with respect to the origin.
    # We fix the angles theta_i = 2*pi*i/N.
    
    K = 3 # Number of Fourier modes
    n_params = 2 * K
    
    # Regularization matrix: Banded SPD
    # We want to penalize higher frequency modes more strongly to keep the shape smooth/convex
    # R is a (2K x 2K) matrix.
    # We can make it block diagonal with increasing penalties for higher k.
    R = np.zeros((n_params, n_params))
    for k in range(1, K+1):
        penalty = k ** 2 # Higher modes penalized more
        idx_a = 2*(k-1)
        idx_b = 2*(k-1)+1
        R[idx_a, idx_a] = penalty
        R[idx_b, idx_b] = penalty
        # Add small off-diagonal terms to make it strictly banded and non-diagonal if desired, 
        # but diagonal is SPD and banded. Let's add small off-diagonals to satisfy "banded" strictly if needed, 
        # though diagonal is a 1-band matrix.
        if k < K:
            # Couple a_k with a_{k+1} slightly? 
            # Let's stick to diagonal for simplicity and stability, it is a valid banded SPD matrix.
            pass
            
    # Ensure R is SPD (it is, diagonal with positive entries)
    
    def objective(params):
        a = params[:K]
        b = params[K:]
        
        # Regularization term
        reg = np.dot(params, R @ params)
        
        # Compute radii
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = np.ones(N)
        for k in range(1, K+1):
            r += a[k-1] * np.cos(k * theta)
            r += b[k-1] * np.sin(k * theta)
            
        # Check if radii are positive
        if np.any(r <= 0):
            return 1e6 + reg
            
        # Compute points
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack((x, y))
        
        # Check convexity
        # For a star-shaped polygon with sorted angles, we check if the cross product of consecutive edge vectors is positive
        # Edge i: P_{i+1} - P_i
        # Cross product Z-component: (x_{i+1}-x_i)(y_{i+2}-y_{i+1}) - (y_{i+1}-y_i)(x_{i+2}-x_{i+1})
        # We need all to have the same sign (positive for CCW).
        # Note: indices mod N
        cross_prods = []
        for i in range(N):
            i1 = i
            i2 = (i + 1) % N
            i3 = (i + 2) % N
            x1, y1 = pts[i1]
            x2, y2 = pts[i2]
            x3, y3 = pts[i3]
            cp = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            cross_prods.append(cp)
            
        cross_prods = np.array(cross_prods)
        if np.any(cross_prods <= 0):
            # Not convex or collinear
            return 1e6 + reg
            
        # Calculate Hull Area
        # Since it's convex and star-shaped, area is 0.5 * sum(r_i^2 * sin(delta_theta))
        # delta_theta = 2*pi/N
        delta_theta = 2 * np.pi / N
        hull_area = 0.5 * np.sum(r**2) * np.sin(delta_theta)
        
        # Calculate Min Triangle Area
        min_area = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            p1 = pts[i]
            p2 = pts[j]
            p3 = pts[k]
            area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if area < min_area:
                min_area = area
                
        score = min_area / hull_area
        # We want to maximize score, so minimize -score
        # Add regularization to the loss to prefer smoother shapes (lower energy in R)
        # The regularization should be scaled appropriately
        return -score + 0.01 * reg

    # Initial guess: Regular polygon (a=0, b=0)
    x0 = np.zeros(n_params)
    
    # Optimize
    # Use L-BFGS-B for speed and robustness
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000})
    
    # Extract best params
    best_params = res.x
    a = best_params[:K]
    b = best_params[K:]
    
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = np.ones(N)
    for k in range(1, K+1):
        r += a[k-1] * np.cos(k * theta)
        r += b[k-1] * np.sin(k * theta)
        
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    
    # Normalize to unit area? 
    # The score is invariant to scale, but the task says "normalized to unit area".
    # The evaluator normalizes by convex hull area.
    # So we just return the points.
    
    return pts
