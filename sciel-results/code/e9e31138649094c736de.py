import numpy as np
import time
from scipy.optimize import root, differential_evolution
import itertools

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _min_area_cart(pts):
    """
    Calculate minimum triangle area for points defined by Cartesian coordinates.
    pts: (N, 2) array.
    """
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    
    # Vectorized cross product (2D)
    # Area = 0.5 * | (pi - pk) x (pj - pk) |
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _solve_kkt_system(x0, time_limit=10.0):
    """
    Solve the KKT system for the Heilbronn problem.
    Variables:
    - p_i (x, y) for i in 0..N-1 (2N vars)
    - A (min area) (1 var)
    - lambda_k for each triple k (165 vars)
    Total vars: 22 + 1 + 165 = 188
    
    Equations:
    1. For each point i (2N eqs):
       Sum_k (lambda_k * dA_k/dp_i) = 0
       Where dA_k/dp_i is the gradient of the signed area of triangle k with respect to point i.
       Note: We use signed area for gradients to handle the non-smoothness of |.| locally,
       assuming the active set (where area == A) determines the sign.
       
    2. For each triple k (165 eqs):
       (A_k - A) * lambda_k = 0
       Where A_k is the signed area of triangle k.
       
    3. Sum of lambda_k = 1 (1 eq)
       This is a gauge fixing to prevent scaling ambiguity in multipliers.
       
    Total equations: 22 + 165 + 1 = 188. Matches number of variables.
    """
    n_vars = 2 * N + 1 + _N_TRIPLES
    n_eqs = 2 * N + _N_TRIPLES + 1
    
    def system(z):
        # Unpack
        p_flat = z[:2 * N]
        A = z[2 * N]
        lambdas = z[2 * N + 1:]
        
        pts = p_flat.reshape(N, 2)
        
        # Check time
        if time.time() > time_limit:
            return np.ones(n_eqs)
            
        # Calculate signed areas for all triples
        pi = pts[_triples[:, 0]]
        pj = pts[_triples[:, 1]]
        pk = pts[_triples[:, 2]]
        
        dx1 = pi[:, 0] - pk[:, 0]
        dy1 = pi[:, 1] - pk[:, 1]
        dx2 = pj[:, 0] - pk[:, 0]
        dy2 = pj[:, 1] - pk[:, 1]
        
        signed_areas = dx1 * dy2 - dy1 * dx2
        abs_areas = np.abs(signed_areas)
        
        # Gradients of signed area with respect to each point
        # dA/dp_i = 0.5 * (cross product terms)
        # For triangle (i, j, k):
        # A = 0.5 * ( (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k) )
        # dA/dx_i = 0.5 * (y_j - y_k)
        # dA/dy_i = 0.5 * (x_k - x_j)
        # dA/dx_j = 0.5 * (y_k - y_i)
        # dA/dy_j = 0.5 * (x_i - x_k)
        # dA/dx_k = 0.5 * (y_i - y_j)
        # dA/dy_k = 0.5 * (x_j - x_i)
        
        # We need Sum_k lambda_k * grad_k for each point.
        # Initialize gradient sums
        grad_sum = np.zeros((N, 2))
        
        # Iterate over triples to accumulate gradients
        # This is O(N_TRIPLES * N) which is 165 * 11 ~ 1800 ops, very fast.
        for idx in range(_N_TRIPLES):
            lam = lambdas[idx]
            if lam == 0:
                continue
            i, j, k = _triples[idx]
            pi_p = pts[i]
            pj_p = pts[j]
            pk_p = pts[k]
            
            # Gradients for i
            grad_i_x = 0.5 * (pj_p[1] - pk_p[1])
            grad_i_y = 0.5 * (pk_p[0] - pj_p[0])
            grad_sum[i, 0] += lam * grad_i_x
            grad_sum[i, 1] += lam * grad_i_y
            
            # Gradients for j
            grad_j_x = 0.5 * (pk_p[1] - pi_p[1])
            grad_j_y = 0.5 * (pi_p[0] - pk_p[0])
            grad_sum[j, 0] += lam * grad_j_x
            grad_sum[j, 1] += lam * grad_j_y
            
            # Gradients for k
            grad_k_x = 0.5 * (pi_p[1] - pj_p[1])
            grad_k_y = 0.5 * (pj_p[0] - pi_p[0])
            grad_sum[k, 0] += lam * grad_k_x
            grad_sum[k, 1] += lam * grad_k_y
            
        # Equations 1: Gradient sum = 0
        eqs = np.zeros(n_eqs)
        eqs[:2 * N] = grad_sum.flatten()
        
        # Equations 2: (A_k_signed - A) * lambda_k = 0 ? 
        # Actually, the condition is that if lambda_k > 0, then A_k = A (or -A depending on orientation).
        # The standard complementarity condition is lambda_k * (A_k - A) = 0.
        # However, A_k can be negative. The "area" is |A_k|.
        # If we assume the optimal configuration has all triangles oriented the same way (or we handle signs),
        # we can work with signed areas. 
        # Let's assume the "active" triangles have signed area = S (where S = +/- A).
        # To be robust, we use the condition: lambda_k * (signed_area_k^2 - A^2) = 0? No, that's not differentiable well.
        
        # Better: lambda_k * (signed_area_k - A) = 0 AND lambda_k * (signed_area_k + A) = 0? No.
        # The condition is that the gradient of the *minimum* area is zero.
        # min_k |A_k|. 
        # If A_k > 0, d|A_k|/dp = dA_k/dp.
        # If A_k < 0, d|A_k|/dp = -dA_k/dp.
        # So the gradient of the objective is Sum_{k in Active} sign(A_k) * lambda_k * grad_A_k = 0.
        # And for k not in Active, lambda_k = 0.
        
        # Let's refine the system:
        # 1. Sum_k sign(A_k) * lambda_k * grad_A_k = 0 (for each point)
        # 2. lambda_k * (A_k - A) = 0 if A_k > 0
        #    lambda_k * (A_k + A) = 0 if A_k < 0
        #    This is equivalent to lambda_k * (A_k^2 - A^2) = 0? No.
        #    It's equivalent to lambda_k * (A_k - A * sign(A_k)) = 0.
        #    But sign(A_k) is discontinuous.
        
        # Alternative: Use the fact that at the optimum, the active set defines the minimum.
        # Let's assume all active triangles have the same orientation (say, positive).
        # If the optimal solution has mixed orientations, we can reflect points.
        # Let's try: lambda_k * (signed_area_k - A) = 0.
        # And add a constraint that A > 0.
        # Also, we need to ensure that for inactive triangles, |signed_area_k| >= A.
        # The root finder will find a stationary point. We must verify it is a local max/min.
        
        # Let's use: eq_k = lambda_k * (signed_area_k - A)
        eqs[2 * N : 2 * N + _N_TRIPLES] = lambdas * (signed_areas - A)
        
        # Equation 3: Sum lambda = 1
        eqs[-1] = np.sum(lambdas) - 1.0
        
        return eqs

    # Initial guess for lambdas: uniform over active set?
    # We don't know active set.
    # Let's try to solve.
    try:
        res = root(system, x0, method='hybr', options={'xtol': 1e-8, 'maxfev': 2000})
        return res.x, res.success
    except Exception:
        return None, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def _is_time_up():
        return time.time() > deadline
        
    best_pts = None
    best_area = 0.0
    
    # 1. Global Search with DE to find good initial guesses for KKT
    # We use DE to maximize min area, but we use the result as a seed for KKT.
    # DE bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    def de_objective(x_flat):
        if _is_time_up():
            return 1e9
        pts = x_flat.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Soft penalty for outside
        d1 = np.maximum(0, -y)
        d2 = np.maximum(0, y - SQRT3 * x)
        d3 = np.maximum(0, y + SQRT3 * x - SQRT3)
        penalty = 50.0 * (np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2))
        
        area = _min_area_cart(pts)
        return -area + penalty

    # Run DE briefly to get a few good candidates
    de_candidates = []
    try:
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol'
        )
        if res_de.success:
            de_candidates.append(res_de.x)
    except Exception:
        pass
        
    # Add some structured inits
    # 1. Centroid + Rings
    # 2. Grid
    
    # Simple structured init: 1 center, 3 inner, 3 mid, 4 outer?
    # For N=11, a known good config is 1 center, 3 on inner triangle, 3 on middle, 3 on outer, 1 more?
    # Let's just use random valid points as extra seeds.
    rng = np.random.default_rng(42)
    for _ in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        pts = np.column_stack([x, y])
        de_candidates.append(pts.flatten())
        
    # 2. Multi-start KKT Solving
    # For each candidate, we need to construct the initial guess for the KKT system.
    # z = [p_flat, A, lambdas]
    # A = min_area(pts)
    # lambdas: We need to guess which triangles are active.
    # Heuristic: Active triangles are those with area close to A.
    # Let's identify active set: |A_k - A| < epsilon.
    
    for idx, x0_flat in enumerate(de_candidates):
        if _is_time_up():
            break
            
        pts = x0_flat.reshape(N, 2)
        
        # Ensure pts are inside
        x = pts[:, 0]
        y = pts[:, 1]
        d1 = np.maximum(0, -y)
        d2 = np.maximum(0, y - SQRT3 * x)
        d3 = np.maximum(0, y + SQRT3 * x - SQRT3)
        
        # If significantly outside, skip or project? 
        # DE with penalty should keep them roughly inside.
        # Let's project to be safe for the KKT solver.
        # Simple projection: clip y
        y = np.maximum(0, y)
        y = np.minimum(y, SQRT3 * x)
        y = np.minimum(y, -SQRT3 * x + SQRT3)
        pts[:, 1] = y
        
        A_init = _min_area_cart(pts)
        
        # Calculate signed areas to determine active set
        pi = pts[_triples[:, 0]]
        pj = pts[_triples[:, 1]]
        pk = pts[_triples[:, 2]]
        dx1 = pi[:, 0] - pk[:, 0]
        dy1 = pi[:, 1] - pk[:, 1]
        dx2 = pj[:, 0] - pk[:, 0]
        dy2 = pj[:, 1] - pk[:, 1]
        signed_areas = dx1 * dy2 - dy1 * dx2
        abs_areas = np.abs(signed_areas)
        
        # Identify active triangles: those with area close to min
        # Use a small tolerance relative to A
        tol = 1e-3 * A_init if A_init > 0 else 1e-6
        active_mask = np.abs(abs_areas - A_init) < tol
        
        # If no active triangles found (shouldn't happen), take the min one
        if not np.any(active_mask):
            min_idx = np.argmin(abs_areas)
            active_mask[min_idx] = True
            
        # Initialize lambdas
        lambdas = np.zeros(_N_TRIPLES)
        # Distribute weight among active triangles
        # Simple heuristic: uniform
        n_active = np.sum(active_mask)
        lambdas[active_mask] = 1.0 / n_active
        
        # Construct z0
        z0 = np.zeros(2 * N + 1 + _N_TRIPLES)
        z0[:2 * N] = pts.flatten()
        z0[2 * N] = A_init
        z0[2 * N + 1:] = lambdas
        
        # Solve KKT
        res_kkt, success = _solve_kkt_system(z0, time_limit=deadline - time.time() - 5)
        
        if success and res_kkt is not None:
            p_sol = res_kkt[:2 * N].reshape(N, 2)
            A_sol = res_kkt[2 * N]
            
            # Verify validity
            x_sol = p_sol[:, 0]
            y_sol = p_sol[:, 1]
            valid = True
            if np.any(y_sol < -1e-6) or np.any(y_sol > SQRT3 * x_sol + 1e-6) or np.any(y_sol > -SQRT3 * x_sol + SQRT3 + 1e-6):
                valid = False
                
            if valid:
                # Calculate actual min area
                actual_area = _min_area_cart(p_sol)
                # Check if the solution is consistent: A_sol should be close to actual_area
                # And lambdas should be non-negative (KKT multipliers)
                lambdas_sol = res_kkt[2 * N + 1:]
                if np.all(lambdas_sol >= -1e-6):
                    if actual_area > best_area:
                        best_area = actual_area
                        best_pts = p_sol.copy()
                        
    # Fallback if KKT didn't find anything better than DE
    if best_pts is None:
        # Use the best DE candidate
        if de_candidates:
            # Find best DE candidate
            best_de_idx = 0
            best_de_area = -1
            for i, x_flat in enumerate(de_candidates):
                pts = x_flat.reshape(N, 2)
                x = pts[:, 0]
                y = pts[:, 1]
                # Project
                y = np.maximum(0, y)
                y = np.minimum(y, SQRT3 * x)
                y = np.minimum(y, -SQRT3 * x + SQRT3)
                pts[:, 1] = y
                area = _min_area_cart(pts)
                if area > best_de_area:
                    best_de_area = area
                    best_de_idx = i
            best_pts = de_candidates[best_de_idx].reshape(N, 2)
            best_area = best_de_area
            
    # Final safety check and clamp
    if best_pts is None:
        # Absolute fallback: Centroid
        best_pts = np.zeros((N, 2))
        best_pts[0] = [0.5, H/3] # Centroid
        
    x = best_pts[:, 0]
    y = best_pts[:, 1]
    y = np.maximum(0, y)
    y = np.minimum(y, SQRT3 * x)
    y = np.minimum(y, -SQRT3 * x + SQRT3)
    best_pts[:, 1] = y
    
    return best_pts
