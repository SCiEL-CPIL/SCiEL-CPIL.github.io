import numpy as np
import time
import itertools

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

# Area of the container triangle
AREA_CONTAINER = np.sqrt(3) / 4.0

# Precompute all triplets
triplets = list(itertools.combinations(range(N), 3))
n_triplets = len(triplets)

def barycentric_coords(p):
    """
    Convert Cartesian point p to barycentric coordinates (u, v, w)
    relative to triangle A, B, C.
    """
    # p = u*A + v*B + w*C, u+v+w=1
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # x = v + 0.5w
    # y = sqrt(3)/2 * w
    w = 2.0 * p[1] / np.sqrt(3)
    v = p[0] - 0.5 * w
    u = 1.0 - v - w
    return u, v, w

def project_to_triangle(p):
    """
    Project point p into the triangle using barycentric clipping.
    """
    u, v, w = barycentric_coords(p)
    
    # Simple clipping
    # If any coord is negative, set to 0 and renormalize others
    # This is an approximation of projection, but keeps points inside.
    # A more accurate projection would be needed for exact nearest point,
    # but this is sufficient for optimization.
    
    if u < 0:
        u = 0
        s = v + w
        if s > 0:
            v /= s
            w /= s
        else:
            # Fallback to centroid
            return (A + B + C) / 3.0
    elif v < 0:
        v = 0
        s = u + w
        if s > 0:
            u /= s
            w /= s
        else:
            return (A + B + C) / 3.0
    elif w < 0:
        w = 0
        s = u + v
        if s > 0:
            u /= s
            v /= s
        else:
            return (A + B + C) / 3.0
            
    # Convert back to Cartesian
    x = v * B[0] + w * C[0] # A is 0
    y = v * B[1] + w * C[1]
    return np.array([x, y])

def compute_areas_and_gradients(points):
    """
    Compute areas and gradients for all triplets.
    Returns:
    areas: array of shape (n_triplets,)
    grads: array of shape (n_triplets, N, 2) - gradient of area w.r.t each point
    """
    areas = np.zeros(n_triplets)
    grads = np.zeros((n_triplets, N, 2))
    
    for idx, (i, j, k) in enumerate(triplets):
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Signed area * 2 = det
        # Area = 0.5 * | (pj - pi) x (pk - pi) |
        # Let's use the formula: 0.5 * | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) |
        
        # Gradient of signed area S = 0.5 * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j))
        # dS/dx_i = 0.5 * (y_j - y_k)
        # dS/dy_i = 0.5 * (x_k - x_j)
        # dS/dx_j = 0.5 * (y_k - y_i)
        # dS/dy_j = 0.5 * (x_i - x_k)
        # dS/dx_k = 0.5 * (y_i - y_j)
        # dS/dy_k = 0.5 * (x_j - x_i)
        
        S = 0.5 * (pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))
        
        # Gradient of |S| is sign(S) * grad(S)
        sign = 1.0 if S >= 0 else -1.0
        if S == 0:
            sign = 1.0 # Arbitrary, gradient is 0 anyway if points coincide? No, grad S is not 0.
            # But if S=0, the function is not differentiable.
            # We just pick a direction.
            
        g_i = np.array([0.5 * (pj[1] - pk[1]), 0.5 * (pk[0] - pj[0])])
        g_j = np.array([0.5 * (pk[1] - pi[1]), 0.5 * (pi[0] - pk[0])])
        g_k = np.array([0.5 * (pi[1] - pj[1]), 0.5 * (pj[0] - pi[0])])
        
        areas[idx] = abs(S)
        
        grads[idx, i] = sign * g_i
        grads[idx, j] = sign * g_j
        grads[idx, k] = sign * g_k
        
    return areas, grads

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Initial points
    # 3 vertices, 3 midpoints, 1 centroid, 4 others
    pts = np.array([
        A, B, C,
        (A + B) / 2.0, (B + C) / 2.0, (C + A) / 2.0,
        (A + B + C) / 3.0,
        # 4 random points near center
        (A + B + C) / 3.0 + np.array([0.1, 0.05]),
        (A + B + C) / 3.0 + np.array([-0.1, 0.05]),
        (A + B + C) / 3.0 + np.array([0.0, -0.1]),
        (A + B + C) / 3.0 + np.array([0.05, 0.05])
    ])
    
    # Project initial points
    for i in range(N):
        pts[i] = project_to_triangle(pts[i])
    
    best_score = -1.0
    best_pts = pts.copy()
    
    m = 100.0 # Sharpness for log-sum-exp
    eta = 0.01 # Learning rate
    
    # Optimization loop
    while time.time() - start_time < time_limit:
        areas, grads = compute_areas_and_gradients(pts)
        
        # Weights for log-sum-exp
        # w_k = exp(-m * A_k)
        # To avoid overflow/underflow, subtract max
        max_area = np.max(areas)
        # We want to emphasize small areas.
        # exp(-m * A) is large for small A.
        # Let's shift: exp(-m * (A - min_A))
        min_area = np.min(areas)
        shifted = areas - min_area
        weights = np.exp(-m * shifted)
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            break
            
        normalized_weights = weights / sum_weights
        
        # Total gradient
        # grad_total = sum_k w_k * grad_k
        # grads shape: (n_triplets, N, 2)
        # weights shape: (n_triplets,)
        grad_total = np.tensordot(normalized_weights, grads, axes=(0, 0))
        
        # Update
        pts_new = pts + eta * grad_total
        
        # Project
        for i in range(N):
            pts_new[i] = project_to_triangle(pts_new[i])
        
        pts = pts_new
        
        # Evaluate current score
        current_min_area = np.min(areas)
        current_score = current_min_area / AREA_CONTAINER
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            
        # Adaptive step size?
        # If score decreases, reduce eta?
        # Simple: keep eta fixed or slowly decay.
        # Let's decay eta slightly
        eta *= 0.999
        
        # Check for stagnation
        if eta < 1e-6:
            break
            
    # Final check
    final_areas, _ = compute_areas_and_gradients(best_pts)
    final_score = np.min(final_areas) / AREA_CONTAINER
    
    if final_score > best_score:
        best_score = final_score
        
    return best_pts

# EVOLVE-BLOCK-END
