import numpy as np
import scipy.optimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    def smooth_objective(params, beta):
        P = params.reshape(N, D)
        # Normalize
        centroid = np.mean(P, axis=0)
        P = P - centroid
        norm_factor = np.sqrt(np.sum(np.sum(P**2, axis=1)) / N)
        if norm_factor < 1e-10:
            return 1e10, None
        P = P / norm_factor
        
        # Pairwise squared distances
        # P is (N, D)
        # dist2[i, j] = sum((P[i]-P[j])**2)
        # Using broadcasting
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = dist2[mask]
        
        # Smooth max and min
        # M = (1/beta) * log(sum(exp(beta * dist2)))
        # m = -(1/beta) * log(sum(exp(-beta * dist2)))
        
        # To avoid overflow, use log-sum-exp trick
        max_d2 = np.max(dist2_masked)
        min_d2 = np.min(dist2_masked)
        
        # If beta is small, just use mean? No, stick to log-sum-exp
        # log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
        
        log_sum_exp_pos = max_d2 + np.log(np.sum(np.exp(beta * (dist2_masked - max_d2))))
        M = log_sum_exp_pos / beta
        
        log_sum_exp_neg = min_d2 - np.log(np.sum(np.exp(-beta * (dist2_masked - min_d2))))
        # Wait, log(sum(exp(-beta * x)))
        # Let y = -beta * x. max_y = -beta * min_x.
        # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
        # = -beta * min_x + log(sum(exp(-beta * (x - min_x))))
        # So sum(exp(-beta * x)) = exp(-beta * min_x) * sum(exp(-beta * (x - min_x)))
        # log(sum(exp(-beta * x))) = -beta * min_x + log(sum(exp(-beta * (x - min_x))))
        
        log_sum_exp_neg_val = -beta * min_d2 + np.log(np.sum(np.exp(-beta * (dist2_masked - min_d2))))
        m = -log_sum_exp_neg_val / beta
        
        if m < 1e-10:
            return 1e10, None
            
        val = (M / m) ** 2
        return val, None

    def objective_wrapper(params):
        # Use a fixed beta for the wrapper, or global?
        # We will update beta externally
        return smooth_objective(params, current_beta[0])[0]

    current_beta = [1.0]
    
    best_score = float('inf')
    best_params = None
    
    # Initial random points
    np.random.seed(42)
    
    def _make_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        return vertices

    def _make_fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def _make_stella_octangula():
        # Two tetrahedra
        # Tetra 1: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
        # Tetra 2: (1,1,-1), (1,-1,1), (-1,1,1), (-1,-1,-1)
        t1 = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ])
        t2 = np.array([
            [1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, -1]
        ])
        points = np.vstack([t1, t2])
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    initializations = []
    
    # 1. Random
    initializations.append(np.random.randn(N, D))
    
    # 2. Icosahedron + 2 random points
    ico = _make_icosahedron()
    extra = np.random.randn(2, D)
    init_ico = np.vstack([ico, extra])
    initializations.append(init_ico)
    
    # 3. Fibonacci Sphere
    init_fib = _make_fibonacci_sphere(N)
    initializations.append(init_fib)
    
    # 4. Stella Octangula + 6 random
    so = _make_stella_octangula()
    extra6 = np.random.randn(6, D)
    init_so = np.vstack([so, extra6])
    initializations.append(init_so)
    
    # 5. Icosahedron + 2 points near center (scaled)
    ico2 = _make_icosahedron()
    extra_center = np.random.randn(2, D) * 0.1
    init_ico_c = np.vstack([ico2, extra_center])
    initializations.append(init_ico_c)

    num_restarts = len(initializations)
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        P_init = initializations[i]
        params_init = P_init.flatten()
        
        # Annealing beta
        betas = [1.0, 10.0, 100.0, 1000.0]
        
        current_params = params_init.copy()
        
        for beta in betas:
            if time.time() - start_time > timeout:
                break
                
            current_beta[0] = beta
            
            # Optimize
            try:
                res = scipy.optimize.minimize(
                    objective_wrapper, 
                    current_params, 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = res.x
            except Exception:
                pass
        
        # Evaluate final score with high beta (or exact)
        # For final score, we should use the exact max/min ratio
        P_final = current_params.reshape(N, D)
        centroid = np.mean(P_final, axis=0)
        P_final = P_final - centroid
        norm_factor = np.sqrt(np.sum(np.sum(P_final**2, axis=1)) / N)
        if norm_factor > 1e-10:
            P_final = P_final / norm_factor
            
        diff = P_final[:, np.newaxis, :] - P_final[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dist2[mask])
        
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        if min_d > 1e-10:
            score = (max_d / min_d) ** 2
            if score < best_score:
                best_score = score
                best_params = P_final.copy()
    
    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_params

# EVOLVE-BLOCK-START
# The code above is the new design.
# I will place it inside the block.
# EVOLVE-BLOCK-END
