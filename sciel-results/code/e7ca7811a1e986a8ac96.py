import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.RandomState(123)
    
    # Initialize barycentric coordinates
    # Start with a grid-like distribution in the simplex
    # We can generate a grid of points in the 2D simplex and pick N
    # Or just random.
    
    # Let's use a "repulsion" init or just random.
    # Random init in simplex:
    # Generate N points uniformly in the simplex.
    # Method: generate N+1 points uniformly in R^3, take the first N, normalize?
    # No, standard way:
    # u, v ~ U(0,1). If u+v > 1, u=1-u, v=1-v.
    # Lambda = [1-u-v, u, v]
    
    def random_barycentric():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        return np.array([1 - u - v, u, v])
        
    # Better init: Use a grid
    # Let's create a 5x5 grid in u,v space
    grid_u = np.linspace(0, 1, 5)
    grid_v = np.linspace(0, 1, 5)
    candidates = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 1:
                candidates.append([1 - u - v, u, v])
                
    # We have 15 candidates. We need 11.
    # Pick 11 that are well spread?
    # Just take the first 11?
    # Or randomize the order and take 11.
    rng.shuffle(candidates)
    pts_bary = np.array(candidates[:N])
    
    # Calculate initial score
    def calc_min_det(pts_bary):
        min_det = float('inf')
        n = len(pts_bary)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Matrix of 3 barycentric vectors
                    # det
                    m = np.array([pts_bary[i], pts_bary[j], pts_bary[k]])
                    d = abs(np.linalg.det(m))
                    if d < min_det:
                        min_det = d
        return min_det
        
    current_score = calc_min_det(pts_bary)
    best_pts = pts_bary.copy()
    best_score = current_score
    
    T = 0.005
    T_min = 1e-7
    cooling = 0.9995
    step_size = 0.05
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        i = rng.randint(N)
        
        # Move
        # Add noise to barycentric coords
        noise = rng.normal(0, step_size, 3)
        new_lambda = pts_bary[i] + noise
        
        # Clip and normalize
        new_lambda = np.clip(new_lambda, 0, None)
        s = np.sum(new_lambda)
        if s == 0:
            # Revert
            continue
        new_lambda /= s
        
        # Calculate score
        old_lambda = pts_bary[i]
        pts_bary[i] = new_lambda
        new_score = calc_min_det(pts_bary)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts_bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            pts_bary[i] = old_lambda
            no_improve_count += 1
            
        # Adapt step size
        if no_improve_count > max_no_improve:
            # Restart or increase step?
            # Let's restart from best with slightly higher T
            pts_bary = best_pts.copy()
            current_score = best_score
            T = max(T_min, T * 0.5) # Cool down more? Or reheat?
            # Reheat to escape local minimum
            T = 0.005 
            step_size = 0.05
            no_improve_count = 0
            max_no_improve = 1000
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert best barycentric to Cartesian
    # P = L1*A + L2*B + L3*C
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # x = L2 + 0.5*L3
    # y = sqrt(3)/2 * L3
    
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts_cart = np.zeros((N, 2))
    for i in range(N):
        l = best_pts[i]
        pts_cart[i] = l[0]*A + l[1]*B + l[2]*C
        
    return pts_cart
