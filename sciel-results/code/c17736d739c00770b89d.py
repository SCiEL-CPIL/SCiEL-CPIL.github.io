import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # 1. Generate initial guess: Regular Octagon
    # Vertices of regular octagon on unit circle
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    # Rotate so that one vertex is at (1,0) and opposite at (-1,0)
    # Actually, standard linspace starts at 0.
    # P0 = (1,0), P4 = (-1,0).
    # Translate to center at (0,0) is already done.
    # We want diameter 2.
    # Fix P0 at (0,0) and P4 at (2,0).
    # Current P0=(1,0), P4=(-1,0).
    # Translate by (-1, 0) -> P0=(0,0), P4=(-2,0).
    # Flip x -> P0=(0,0), P4=(2,0).
    
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Identify indices of opposite points
    # With linspace, 0 and 4 are opposite.
    # Let's just fix index 0 and 4.
    p0 = pts[0]
    p4 = pts[4]
    
    # Translate and scale/rotate so p0 becomes (0,0) and p4 becomes (2,0)
    # Vector v = p4 - p0
    v = p4 - p0
    dist = np.linalg.norm(v)
    # v is (-2, 0) in standard orientation? 
    # cos(0)=1, sin(0)=0 -> (1,0)
    # cos(pi)=-1, sin(pi)=0 -> (-1,0)
    # v = (-2, 0).
    # We want to map v to (2,0).
    # So we need to rotate by 180 degrees (or -180) and translate.
    
    # Simpler: Just define the 8 points explicitly for regular octagon with diameter 2.
    # Points:
    # (0,0)
    # (2,0)
    # The other 6 points are determined by the regular octagon geometry.
    # Side length s = 2 * sin(pi/8).
    # Coordinates relative to center (1,0) with radius 1:
    # (1,0), (1+cos(pi/4), sin(pi/4)), (1+cos(pi/2), sin(pi/2)) ... wait.
    
    # Let's just use the generated points, fix 0 and 4, and let the optimizer move the rest.
    # We need to transform the initial points such that pts[0] = (0,0) and pts[4] = (2,0).
    
    # Current pts[0] = (1,0), pts[4] = (-1,0).
    # Transform: T(x) = -x + (1,0)? 
    # T(1,0) = -1+1, 0 = (0,0).
    # T(-1,0) = -(-1)+1, 0 = (2,0).
    # So new_pts = -pts + [1, 0].
    
    init_pts = -pts + np.array([1.0, 0.0])
    
    # Verify
    # init_pts[0] = -(1,0) + (1,0) = (0,0).
    # init_pts[4] = -(-1,0) + (1,0) = (2,0).
    
    # Free variables: indices 1, 2, 3, 5, 6, 7.
    fixed_idx = [0, 4]
    free_idx = [i for i in range(8) if i not in fixed_idx]
    
    def unpack(x):
        # x is 12-dim vector
        p = np.zeros((8, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2*i], x[2*i+1]]
        return p

    def objective(x):
        p = unpack(x)
        # Calculate log product of distances
        log_prod = 0.0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(p[i] - p[j])
                if d < 1e-9:
                    return -1e10 # Avoid log(0)
                log_prod += np.log(d)
        return -log_prod # Minimize negative log product

    def constraints(x):
        p = unpack(x)
        cons = []
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(p[i] - p[j])
                # Constraint: d <= 2
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 2.0 - np.linalg.norm(unpack(x)[i] - unpack(x)[j])})
        return cons

    # SLSQP can be slow with many constraints. 
    # Alternative: Penalty method with L-BFGS-B.
    # Let's try SLSQP first as it's more robust for small problems.
    
    x0 = np.array([init_pts[i] for i in free_idx]).flatten()
    
    best_x = x0
    best_score = objective(x0)
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success:
            if res.fun < best_score:
                best_x = res.x
                best_score = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > timeout - 10:
        # Return best found so far
        p = unpack(best_x)
        return p.tolist()

    # Multi-start to ensure we don't get stuck in a bad local minimum
    # The regular octagon is a strong candidate, but we can try perturbing it.
    
    np.random.seed(42)
    
    for _ in range(10):
        if time.time() - start_time > timeout - 20:
            break
            
        # Perturb initial points
        noise = np.random.normal(0, 0.1, size=(8, 2))
        # Keep fixed points fixed
        noise[0] = 0
        noise[4] = 0
        perturbed = init_pts + noise
        
        x0_pert = np.array([perturbed[i] for i in free_idx]).flatten()
        
        try:
            res = minimize(objective, x0_pert, method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
        except Exception:
            continue

    p = unpack(best_x)
    
    # Scale check: The optimizer ensures d <= 2. 
    # The score is defined after scaling to diameter 2.
    # If the max distance is < 2, we should scale up to diameter 2 to maximize the product.
    # However, the constraint d <= 2 allows d < 2.
    # The objective function maximizes the product for the *current* scale.
    # If the optimal solution has max distance < 2, scaling it up to diameter 2 would increase all distances, thus increasing the product.
    # Therefore, the global optimum MUST have max distance exactly 2.
    # So the constraint d <= 2 will be active for at least one pair.
    
    # Just in case, let's verify the diameter and scale if necessary (though it should be 2).
    max_d = 0
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d
    
    if max_d > 0:
        scale_factor = 2.0 / max_d
        p = p * scale_factor
    
    # Ensure coordinates are floats and within limits
    return p.tolist()
