"""
Heilbronn triangles: Place 11 points to maximize the minimum triangle area.
Strategy: Multi-start SLSQP optimization with vectorized NumPy evaluation.
Uses a smooth log-sum-exp approximation of the minimum area to provide gradients.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 120s limit

def _vectorized_min_area(pts):
    """
    Compute the minimum area of all triangles formed by the points.
    pts: (N, 2) array of points.
    Returns the raw minimum area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute all pairwise differences: (N, N, 2)
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute cross products for all pairs (i, j)
    # cross[i, j] = diff[i, j, 0] * diff[i, j, 1] - diff[i, j, 1] * diff[i, j, 0]
    # Wait, cross product of vector (x1, y1) and (x2, y2) is x1*y2 - y1*x2.
    # Here diff[i,j] is a single vector. We need cross product of two vectors.
    # Let's construct matrices of vectors.
    # For each pair (i, j), we have vector v_ij = p_j - p_i.
    # We want area of triangle (i, j, k).
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|.
    
    # Let's compute the matrix of cross products for all triplets.
    # This might be memory heavy if N is large, but N=11 is small.
    # Number of triplets is 11*10*9 = 990.
    # We can compute all areas in a vectorized way.
    
    # Get all unique triplets (i, j, k) with i < j < k
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    triplets = np.argwhere(mask)
    
    if len(triplets) == 0:
        return 0.0
        
    # Extract points for each triplet
    p1 = pts[triplets[:, 0]]  # (M, 2)
    p2 = pts[triplets[:, 1]]  # (M, 2)
    p3 = pts[triplets[:, 2]]  # (M, 2)
    
    # Compute cross products
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective_and_grad(x, temp=0.05):
    """
    Objective function for SLSQP.
    We want to MAXIMIZE the min area.
    SLSQP minimizes. So we minimize -min_area.
    Since min is non-smooth, we use a smooth approximation:
    f(x) = - (1/temp) * log(sum(exp(-area_i / temp)))
    This approximates -min(area_i) as temp -> 0.
    """
    pts = x.reshape(N, 2)
    
    # Compute areas
    n = N
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    triplets = np.argwhere(mask)
    
    p1 = pts[triplets[:, 0]]
    p2 = pts[triplets[:, 1]]
    p3 = pts[triplets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Smooth minimum
    # To avoid overflow in exp, subtract max
    max_area = np.max(areas)
    exp_vals = np.exp(-(areas - max_area) / temp)
    sum_exp = np.sum(exp_vals)
    
    # f = - (1/temp) * (max_area + log(sum_exp))
    # f = -max_area/temp - log(sum_exp)/temp
    obj_val = -(max_area + temp * np.log(sum_exp)) / temp
    
    # Gradient calculation
    # d/dx [ - (max_area + temp * log(sum_exp)) / temp ]
    # = -1/temp * (d(max_area)/dx + temp * d(log(sum_exp))/dx)
    # This is tricky because max_area changes which index is max.
    # However, the log-sum-exp gradient is well-defined and smooth.
    # Let L = log(sum(exp(-a_i/temp)))
    # dL/dx = sum( exp(-a_i/temp) * (-1/temp * da_i/dx) ) / sum(exp(-a_i/temp))
    #        = -1/temp * sum( w_i * da_i/dx ) where w_i = exp(-a_i/temp) / sum_exp
    # 
    # Objective J = - (max_area + temp * L) / temp = -max_area/temp - L
    # dJ/dx = -1/temp * d(max_area)/dx - dL/dx
    #
    # The term -1/temp * d(max_area)/dx is the gradient of the hard minimum (with sign flip for max).
    # The term -dL/dx is the gradient of the log-sum-exp correction.
    # As temp -> 0, w_i concentrates on the min area, and L approximates -min_area.
    # So -L approximates min_area.
    # Thus J approximates -min_area.
    #
    # For the gradient, we can just use the gradient of the smooth approximation -L.
    # d(-L)/dx = 1/temp * sum( w_i * da_i/dx )
    #
    # Let's compute da_i/dx.
    # area_i = 0.5 * |cross_i|
    # da_i/dx = 0.5 * sign(cross_i) * d(cross_i)/dx
    #
    # cross_i = (p2-p1)_x (p3-p1)_y - (p2-p1)_y (p3-p1)_x
    # Let p1=(x1,y1), p2=(x2,y2), p3=(x3,y3)
    # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # 
    # d(cross)/dx1 = -(y3-y1) - (y2-y1)(-1) = -y3+y1+y2-y1 = y2-y3
    # d(cross)/dy1 = -(x2-x1) - (y2-y1)(0) ... wait.
    # Let's use partial derivatives carefully.
    # f = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # df/dx1 = -1*(y3-y1) - (y2-y1)*(-1) = -y3+y1+y2-y1 = y2-y3
    # df/dy1 = (x2-x1)*(-1) - (y2-y1)*0 = x1-x2
    # df/dx2 = 1*(y3-y1) - 0 = y3-y1
    # df/dy2 = 0 - 1*(x3-x1) = x1-x3
    # df/dx3 = 0 - (y2-y1)*1 = y1-y2
    # df/dy3 = (x2-x1)*1 - 0 = x2-x1
    
    # We need to accumulate gradients for all points.
    # grad_obj = zeros((N, 2))
    # For each triplet i:
    #   w = w_i
    #   sign_c = sign(cross_i)
    #   da/dx = 0.5 * sign_c * d(cross)/dx
    #   dJ/dx = 1/temp * w * da/dx
    
    # Vectorized gradient accumulation
    # w_i shape (M,)
    # cross shape (M,)
    # sign_c = np.sign(cross)
    # 
    # We need to scatter the gradients.
    # For each triplet (i, j, k):
    #   dx1 += 0.5 * sign_c * (y_j - y_k)
    #   dy1 += 0.5 * sign_c * (x_i - x_j)  -- wait, df/dy1 = x1-x2? 
    #   Let's re-verify df/dy1.
    #   f = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    #   d/dy1: (x2-x1)(-1) - (y2-1)(0) = x1-x2. Correct.
    #   d/dy2: 0 - 1*(x3-x1) = x1-x3. Correct.
    #   d/dy3: (x2-x1)(1) - 0 = x2-x1. Correct.
    
    w = exp_vals / sum_exp
    sign_c = np.sign(cross)
    factor = 0.5 * sign_c * w / temp
    
    # Initialize gradient
    grad = np.zeros((N, 2))
    
    # p1 is index i
    # d(cross)/dx1 = y2 - y3
    # d(cross)/dy1 = x1 - x2
    grad[triplets[:, 0], 0] += factor * (p2[:, 1] - p3[:, 1])
    grad[triplets[:, 0], 1] += factor * (p1[:, 0] - p2[:, 0])
    
    # p2 is index j
    # d(cross)/dx2 = y3 - y1
    # d(cross)/dy2 = x1 - x3
    grad[triplets[:, 1], 0] += factor * (p3[:, 1] - p1[:, 1])
    grad[triplets[:, 1], 1] += factor * (p1[:, 0] - p3[:, 0])
    
    # p3 is index k
    # d(cross)/dx3 = y1 - y2
    # d(cross)/dy3 = x2 - x1
    grad[triplets[:, 2], 0] += factor * (p1[:, 1] - p2[:, 1])
    grad[triplets[:, 2], 1] += factor * (p2[:, 0] - p1[:, 0])
    
    return obj_val, grad

def _is_inside_triangle(pts):
    """Check if all points are inside the triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    # y >= 0
    # y <= S3H * x
    # y <= -S3H * x + S3H
    return np.all(y >= -1e-9) and np.all(y <= S3H * x + 1e-9) and np.all(y <= -S3H * x + S3H + 1e-9)

def construct():
    start_time = time.time()
    
    # Initial best points
    best_points = None
    best_score = -np.inf
    
    # Define constraints
    # Inequality constraints: g(x) >= 0
    # 1. y_i >= 0
    # 2. y_i - S3H * x_i <= 0  => S3H * x_i - y_i >= 0
    # 3. y_i + S3H * x_i - S3H <= 0 => S3H - y_i - S3H * x_i >= 0
    
    def constraint1(x):
        pts = x.reshape(N, 2)
        return pts[:, 1]
        
    def constraint2(x):
        pts = x.reshape(N, 2)
        return S3H * pts[:, 0] - pts[:, 1]
        
    def constraint3(x):
        pts = x.reshape(N, 2)
        return S3H - pts[:, 1] - S3H * pts[:, 0]
        
    constraints = [
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}
    ]
    
    # Multi-start
    np.random.seed(42)
    
    # Generate diverse initial points
    # 1. Random
    # 2. Grid-like
    # 3. Vertices + random
    
    initializations = []
    
    # Random
    for _ in range(10):
        u = np.random.random((N, 2))
        # Map to triangle
        # Use barycentric coordinates
        r1 = np.random.random(N)
        r2 = np.random.random(N)
        r1 = np.sqrt(r1)
        # Actually, simpler: uniform in square, reject outside, or use affine transform.
        # Affine transform from unit square to triangle is not uniform.
        # Let's use the rejection method for initialization, or just random and clamp.
        # For SLSQP, we need a valid start.
        # Let's generate random points in the triangle properly.
        # Area of triangle is S3H/2.
        # Uniform sampling:
        # x = u1 + 0.5*u2
        # y = S3H*u2
        # where u1, u2 are such that u1+u2 <= 1? No.
        # Standard method:
        # r1, r2 uniform in [0,1]
        # if r1+r2 > 1: r1=1-r1, r2=1-r2
        # x = r1 + 0.5*r2
        # y = S3H*r2
        
        r1 = np.random.random(N)
        r2 = np.random.random(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        x = r1 + 0.5 * r2
        y = S3H * r2
        initializations.append(np.column_stack((x, y)))
        
    # Grid-like
    # 4x3 grid in barycentric?
    # Let's try a few structured ones
    # 1. Vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, S3H])
    
    # Add random points near vertices
    for _ in range(5):
        pts = [v1, v2, v3]
        for _ in range(8):
            # Random near center
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = S3H * r2
            pts.append([x, y])
        initializations.append(np.array(pts))
        
    # Run SLSQP for each initialization
    for init in initializations:
        if time.time() > DEADLINE:
            break
            
        # Ensure init is valid
        if not _is_inside_triangle(init):
            continue
            
        x0 = init.flatten()
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='SLSQP',
                bounds=[(0, 1)] * (2 * N), # Loose bounds, constraints handle it
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.status == 0:
                pts_res = res.x.reshape(N, 2)
                if _is_inside_triangle(pts_res):
                    score = _vectorized_min_area(pts_res)
                    if score > best_score:
                        best_score = score
                        best_points = pts_res.copy()
                        
        except Exception:
            continue
            
    # If no valid points found, fallback
    if best_points is None:
        # Use the last valid initialization or random
        r1 = np.random.random(N)
        r2 = np.random.random(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        x = r1 + 0.5 * r2
        y = S3H * r2
        best_points = np.column_stack((x, y))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area:", _vectorized_min_area(pts))
