import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)

def get_triangle_indices(n):
    """Pre-compute indices for all unique triplets."""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices)

TRIANGLE_INDICES = get_triangle_indices(N)
NUM_TRIANGLES = TRIANGLE_INDICES.shape[0]

def compute_areas_and_grads(points):
    """
    Compute areas of all triangles and their gradients with respect to points.
    Area = 0.5 * |cross(p2-p1, p3-p1)|
    Gradient of area w.r.t p_i involves cross products.
    """
    n = points.shape[0]
    areas = np.zeros(NUM_TRIANGLES)
    grads = np.zeros((n, 2))
    
    # Vectorized area calculation
    # p1, p2, p3
    p1 = points[TRIANGLE_INDICES[:, 0]]
    p2 = points[TRIANGLE_INDICES[:, 1]]
    p3 = points[TRIANGLE_INDICES[:, 2]]
    
    # Cross product z-component: (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # Area = 0.5 * |cross|
    cross_z = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    abs_cross = np.abs(cross_z)
    areas = 0.5 * abs_cross
    
    # To handle the non-differentiability at 0, we use the sign of cross_z for gradients
    # If cross_z > 0, grad_area = 0.5 * grad(cross_z)
    # If cross_z < 0, grad_area = -0.5 * grad(cross_z)
    # grad(cross_z) w.r.t p1: -((p3-p2)_y, -(p3-p2)_x) -> wait, let's derive properly.
    # A = 0.5 * ( (x2-x1)(y3-y1) - (y2-y1)(x3-x1) )
    # dA/dx1 = 0.5 * ( -1*(y3-y1) + (y2-y1)*1 ) = 0.5 * (y2 - y3)
    # dA/dy1 = 0.5 * ( -(x2-x1) + (x3-x1)*1 ) = 0.5 * (x3 - x2)
    # dA/dx2 = 0.5 * ( 1*(y3-y1) - 0 ) = 0.5 * (y3 - y1)
    # dA/dy2 = 0.5 * ( 0 - 1*(x3-x1) ) = 0.5 * (x1 - x3)
    # dA/dx3 = 0.5 * ( 0 - (y2-y1)*1 ) = 0.5 * (y1 - y2)
    # dA/dy3 = 0.5 * ( 1*(x2-x1) - 0 ) = 0.5 * (x2 - x1)
    
    # These are gradients of the signed area (cross_z / 2).
    # We want gradients of |signed_area|.
    # sign = np.sign(cross_z)
    # If cross_z == 0, gradient is undefined, we can pick a random direction or 0.
    signs = np.sign(cross_z)
    # Avoid 0 signs for stability, though measure zero
    signs[signs == 0] = 1.0
    
    # Gradient of signed area * 0.5
    # g1 = 0.5 * signs * ( [y2-y3, x3-x2], [y3-y1, x1-x3], [y1-y2, x2-x1] )
    
    # We accumulate gradients into the global points array
    # For each triangle t:
    # grad_points[i] += sign_t * 0.5 * [y_j - y_k, x_k - x_j]
    # grad_points[j] += sign_t * 0.5 * [y_k - y_i, x_i - x_k]
    # grad_points[k] += sign_t * 0.5 * [y_i - y_j, x_j - x_i]
    
    # Let's vectorize this accumulation using np.add.at or similar, but simple loops over triangles might be faster for N=11 (165 triangles) if Python overhead is low, 
    # but vectorization is better.
    
    # Indices for points in each triangle
    i_idx = TRIANGLE_INDICES[:, 0]
    j_idx = TRIANGLE_INDICES[:, 1]
    k_idx = TRIANGLE_INDICES[:, 2]
    
    # Gradient components for point i in triangle t
    # gx_i = 0.5 * sign * (y_j - y_k)
    # gy_i = 0.5 * sign * (x_k - x_j)
    
    gy_j = p2[:, 1]
    gy_k = p3[:, 1]
    gx_k = p3[:, 0]
    gx_j = p2[:, 0]
    
    gx_i = 0.5 * signs * (gy_j - gy_k)
    gy_i = 0.5 * signs * (gx_k - gx_j)
    
    gy_k = p3[:, 1]
    gy_i_temp = p1[:, 1]
    gx_i_temp = p1[:, 0]
    gx_k = p3[:, 0]
    
    gx_j = 0.5 * signs * (gy_k - gy_i_temp)
    gy_j = 0.5 * signs * (gx_i_temp - gx_k)
    
    gx_k = 0.5 * signs * (gy_i_temp - gy_j) # Wait, gy_j was overwritten. Let's use original names.
    
    # Let's re-calculate cleanly
    # p1=(x1,y1), p2=(x2,y2), p3=(x3,y3)
    # g_x1 = 0.5 * s * (y2 - y3)
    # g_y1 = 0.5 * s * (x3 - x2)
    # g_x2 = 0.5 * s * (y3 - y1)
    # g_y2 = 0.5 * s * (x1 - x3)
    # g_x3 = 0.5 * s * (y1 - y2)
    # g_y3 = 0.5 * s * (x2 - x1)
    
    s = signs
    y1 = p1[:, 1]; y2 = p2[:, 1]; y3 = p3[:, 1]
    x1 = p1[:, 0]; x2 = p2[:, 0]; x3 = p3[:, 0]
    
    gx1 = 0.5 * s * (y2 - y3)
    gy1 = 0.5 * s * (x3 - x2)
    gx2 = 0.5 * s * (y3 - y1)
    gy2 = 0.5 * s * (x1 - x3)
    gx3 = 0.5 * s * (y1 - y2)
    gy3 = 0.5 * s * (x2 - x1)
    
    # Accumulate
    np.add.at(grads, (i_idx, 0), gx1)
    np.add.at(grads, (i_idx, 1), gy1)
    np.add.at(grads, (j_idx, 0), gx2)
    np.add.at(grads, (j_idx, 1), gy2)
    np.add.at(grads, (k_idx, 0), gx3)
    np.add.at(grads, (k_idx, 1), gy3)
    
    return areas, grads

def boundary_penalty_and_grad(points):
    """
    Calculate penalty for being outside the triangle and its gradient.
    Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    Inequalities:
    1. y >= 0
    2. sqrt(3)*x - y <= sqrt(3)/2  => y >= sqrt(3)*x - sqrt(3)/2 ? No.
       Line from (1,0) to (0.5, h): y = -sqrt(3)(x-1) = -sqrt(3)x + sqrt(3).
       So y <= -sqrt(3)x + sqrt(3)  => sqrt(3)x + y <= sqrt(3)
    3. Line from (0,0) to (0.5, h): y = sqrt(3)x.
       So y <= sqrt(3)x  => -sqrt(3)x + y >= 0 ? No, points are below this line?
       (0.5, sqrt(3)/6) is inside. sqrt(3)*0.5 - sqrt(3)/6 = sqrt(3)/3 > 0.
       Wait, the triangle is bounded by y=0, y=sqrt(3)x, and y=-sqrt(3)x+sqrt(3).
       Let's check a point inside, e.g., (0.5, 0.1).
       y >= 0: 0.1 >= 0 (OK)
       y <= sqrt(3)x: 0.1 <= 1.732*0.5 = 0.866 (OK)
       y <= -sqrt(3)x + sqrt(3): 0.1 <= -0.866 + 1.732 = 0.866 (OK)
       
    So constraints:
    c1 = y
    c2 = sqrt(3)*x - y
    c3 = sqrt(3) - sqrt(3)*x - y
    
    Penalty = sum(max(0, -c)^2)
    Grad:
    if c < 0: grad += -2 * (-c) * grad_c = 2c * grad_c? No.
    P = sum (max(0, -c))^2.
    dP/dc = 2 * max(0, -c) * d(max(0, -c))/dc
    if c < 0: max(0, -c) = -c. d(-c)/dc = -1. dP/dc = 2(-c)(-1) = 2c.
    if c >= 0: P=0, dP/dc=0.
    
    So grad_P = sum_{c<0} 2c * grad_c.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    c1 = y
    c2 = SQRT3 * x - y
    c3 = SQRT3 - SQRT3 * x - y
    
    pen = 0.0
    grad = np.zeros_like(points)
    
    # Helper to add penalty and grad for a constraint c with grad_c
    def add_pen(c, gc_x, gc_y):
        nonlocal pen, grad
        mask = c < 0
        if np.any(mask):
            # P_i = c_i^2 (since max(0, -c)^2 = c^2 when c<0)
            # dP/dc = 2c
            pen += np.sum(c[mask]**2)
            # grad contribution: 2 * c * grad_c
            grad[mask, 0] += 2 * c[mask] * gc_x[mask]
            grad[mask, 1] += 2 * c[mask] * gc_y[mask]
            
    # c1 = y, grad = (0, 1)
    add_pen(c1, np.zeros_like(x), np.ones_like(y))
    
    # c2 = sqrt(3)x - y, grad = (sqrt(3), -1)
    add_pen(c2, np.full_like(x, SQRT3), -np.ones_like(y))
    
    # c3 = sqrt(3) - sqrt(3)x - y, grad = (-sqrt(3), -1)
    add_pen(c3, -np.full_like(x, SQRT3), -np.ones_like(y))
    
    return pen, grad

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_min_area = -np.inf
    
    # Random restarts with custom gradient ascent
    for seed in range(50):
        if time.time() - start_time > time_limit - 5:
            break
            
        rng = np.random.RandomState(seed)
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            points[i, 0] = v * 1.0 + w * 0.5
            points[i, 1] = w * SQRT3 / 2.0
            
        # Custom Optimization Loop
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(points)
        
        # We want to maximize min_area.
        # Objective: f(points) = min(areas) - penalty
        # Gradient of f is tricky due to min.
        # Approximation: Gradient of average of bottom-k areas, or just the min area gradient.
        # Using the gradient of the single minimum area is noisy.
        # Using the gradient of the sum of areas below a threshold is better.
        
        for step in range(2000):
            if time.time() - start_time > time_limit - 5:
                break
                
            areas, grads = compute_areas_and_grads(points)
            min_area = np.min(areas)
            
            # Penalty
            pen, pen_grad = boundary_penalty_and_grad(points)
            
            # If we are violating constraints significantly, prioritize fixing that
            if pen > 1e-6:
                # Move in direction of negative penalty gradient
                grad_step = -pen_grad * 0.1
            else:
                # Maximize min area
                # Identify triangles with area close to min
                # Let's use the gradient of the minimum area triangle
                # To reduce noise, average gradients of triangles with area < min_area + epsilon
                threshold = min_area + 1e-4
                mask = areas < threshold
                if np.sum(mask) > 0:
                    # Sum gradients of these triangles
                    # Note: grads from compute_areas_and_grads is the gradient of the SUM of areas?
                    # No, compute_areas_and_grads returns grads accumulated from ALL triangles.
                    # I need the gradient of the SPECIFIC triangles.
                    # Let's re-compute or store per-triangle grads?
                    # For N=11, 165 triangles. Storing (165, 11, 2) is fine.
                    
                    # Let's modify compute_areas_and_grads to return per-triangle grads?
                    # Or just use the total gradient? No, total gradient includes large areas which don't matter for min.
                    
                    # Let's do a simpler approach:
                    # Use the gradient of the single minimum triangle. It's steeper and focused.
                    min_idx = np.argmin(areas)
                    i, j, k = TRIANGLE_INDICES[min_idx]
                    p1 = points[i]; p2 = points[j]; p3 = points[k]
                    cross_z = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    s = np.sign(cross_z)
                    if s == 0: s = 1.0
                    
                    # Gradient of this specific triangle area
                    g_tri = np.zeros_like(points)
                    g_tri[i, 0] = 0.5 * s * (p2[1] - p3[1])
                    g_tri[i, 1] = 0.5 * s * (p3[0] - p2[0])
                    g_tri[j, 0] = 0.5 * s * (p3[1] - p1[1])
                    g_tri[j, 1] = 0.5 * s * (p1[0] - p3[0])
                    g_tri[k, 0] = 0.5 * s * (p1[1] - p2[1])
                    g_tri[k, 1] = 0.5 * s * (p2[0] - p1[0])
                    
                    grad_step = g_tri
                else:
                    grad_step = np.zeros_like(points)
                
                # Subtract penalty gradient (we want to minimize penalty)
                grad_step -= pen_grad * 10.0

            # Update momentum
            v = momentum * v + (1 - momentum) * grad_step
            points += lr * v
            
            # Project points back into triangle roughly?
            # The penalty gradient should handle it, but if they fly out, reset.
            if np.any(points[:, 1] < -0.5) or np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 1.5):
                # Reset to initial for this seed
                points = np.zeros((N, 2))
                for i in range(N):
                    while True:
                        u = rng.rand()
                        v_rand = rng.rand()
                        if u + v_rand <= 1:
                            break
                    w = 1 - u - v_rand
                    points[i, 0] = v_rand * 1.0 + w * 0.5
                    points[i, 1] = w * SQRT3 / 2.0
                v = np.zeros_like(points)
                break
            
            # Check improvement
            if pen < 1e-4:
                current_score = min_area
                if current_score > best_min_area:
                    best_min_area = current_score
                    best_points = points.copy()
                    
    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
