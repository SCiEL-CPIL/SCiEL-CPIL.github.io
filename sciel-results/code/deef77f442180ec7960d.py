import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (N, 3) to Cartesian coordinates (N, 2).
    bary columns sum to 1.
    P = lambda0 * V0 + lambda1 * V1 + lambda2 * V2
    """
    # V0 is (0,0), so term 0 is 0
    # P_x = 1.0 * lambda1 + 0.5 * lambda2
    # P_y = 0.0 * lambda0 + 0.0 * lambda1 + H * lambda2
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])

def _min_triangle_area_bary(bary):
    """
    Calculate minimum triangle area for points defined by barycentric coordinates.
    bary: (N, 3) array where each row sums to 1.
    """
    pts = _bary_to_cart(bary)
    
    # Extract points for each triple
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    
    # Vectorized cross product (2D)
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(bary_flat):
    """
    Objective function for minimization.
    We want to maximize min_area, so we minimize -min_area.
    Barycentric coordinates are constrained to the simplex [0,1] with sum=1.
    Differential Evolution handles bounds, but we must ensure sum=1.
    Standard DE works on independent variables. 
    To handle the sum=1 constraint in DE, we can optimize over N-3 variables 
    (setting the last 3 to be determined? No, that's complex).
    
    Alternative: Optimize over 2N Cartesian coordinates with a penalty? 
    No, the prompt suggests a different strategy. 
    Let's stick to Cartesian for DE but with a strict penalty or projection?
    Actually, DE works best with box constraints. 
    Let's use the barycentric representation but relax the sum constraint 
    by normalizing inside the objective? No, that distorts the space.
    
    Better approach for DE: 
    Use Cartesian coordinates (2N vars) with bounds [0,1] for x and [0,H] for y.
    Add a heavy penalty for points outside the triangle.
    However, the current program already does this and gets 9.8.
    
    Let's try a different "Materially Different" angle:
    Use **Simulated Annealing** or **Basin Hopping** with a better local optimizer?
    
    Let's look at Inspiration 1: It uses `_local_optimize` and structured inits.
    Inspiration 2: Differential Evolution.
    
    Let's implement a robust Differential Evolution on **Cartesian coordinates** 
    but with a **strict projection** back into the triangle at every step 
    to ensure the landscape is only evaluated on valid points. 
    This avoids the "penalty gradient" issues.
    
    Wait, DE is stochastic. If we project, the function is discontinuous at boundaries.
    But for "max min area", points on the boundary are often optimal.
    
    Let's try **Basin Hopping** with a high-quality local optimizer (Nelder-Mead) 
    but with a **structured initialization** based on the known optimal structure 
    (1 center, 2 rings). The current program uses random inits.
    
    Actually, the most promising "different" strategy is **Barycentric Optimization** 
    using a transformation that maps a hypercube to the simplex, allowing 
    standard gradient-free optimizers to work on the interior.
    
    Let's use the **Dirichlet distribution** property. 
    If we generate $u_i \sim U(0,1)$, then $v_i = u_i^{1/k}$? No.
    
    Let's stick to the most robust global optimizer for non-smooth functions: 
    **Differential Evolution** on **Barycentric coordinates** with a 
    **penalty for non-normalization** is bad.
    
    Let's use **Differential Evolution** on **Cartesian coordinates** 
    but with a **very tight box** and a **soft constraint** that is 
    active only when outside.
    
    Actually, let's look at the score gap. 9.8 is low. The theoretical max is likely higher.
    The issue with Nelder-Mead is getting stuck in local minima of the "min area" function.
    
    **Strategy:** 
    1. Use **Differential Evolution** (DE) for global search.
    2. Variables: 22 Cartesian coordinates.
    3. Bounds: x in [0, 1], y in [0, H].
    4. Objective: -min_area + penalty_outside.
    5. Penalty: Smooth quadratic penalty for violation.
    6. Key difference: DE explores the whole space, avoiding local traps of NM.
    7. Post-processing: Run a few Nelder-Mead polishes on the best DE result.
    """
    pts = bary_flat.reshape(N, 2)
    
    # Check bounds and penalty
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Triangle inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1)  =>  y <= -sqrt(3)x + sqrt(3)
    
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, y - SQRT3 * x)
    d3 = np.maximum(0, y + SQRT3 * x - SQRT3)
    
    # Scale distances to be comparable to area
    # Area scale is ~ 0.43. Distances are ~ 0.5.
    # Penalty should dominate if outside.
    penalty = 100.0 * (np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2))
    
    if penalty > 1e-6:
        # If significantly outside, return large value
        # But we still want to guide it back.
        pass
        
    area = _min_triangle_area_bary(pts) # Wait, _min_triangle_area_bary expects bary. 
    # I defined _min_triangle_area_bary to take bary, but it calls _bary_to_cart.
    # I need a _min_triangle_area_cart or just reuse the logic.
    
    # Let's refactor to have a clean cart area function.
    return -area + penalty

def _min_area_cart(pts):
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin

    def _is_time_up():
        return time.time() > deadline

    # Bounds for DE
    # x: [0, 1], y: [0, H]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    # Objective for DE
    def de_objective(x_flat):
        if _is_time_up():
            return 1e9
        pts = x_flat.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalties for being outside
        d1 = np.maximum(0, -y)
        d2 = np.maximum(0, y - SQRT3 * x)
        d3 = np.maximum(0, y + SQRT3 * x - SQRT3)
        
        # Normalize penalty to be large
        penalty = 50.0 * (np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2))
        
        area = _min_area_cart(pts)
        
        return -area + penalty

    # Run Differential Evolution
    # popsize=15, maxiter=200 should be fast enough and effective for 22D.
    # tol=1e-6
    try:
        res = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True, # Use L-BFGS-B for polish, might be slow if non-diff, but DE handles it
            init='sobol'
        )
    except Exception:
        res = None

    best_pts = None
    best_area = 0.0

    if res is not None and res.success:
        cand = res.x.reshape(N, 2)
        # Verify validity
        x = cand[:, 0]
        y = cand[:, 1]
        if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
            area = _min_area_cart(cand)
            if area > best_area:
                best_area = area
                best_pts = cand.copy()

    # If DE failed or didn't find a valid point, fallback to a structured init + NM
    if best_pts is None or _is_time_up():
        # Fallback: Random valid points
        rng = np.random.default_rng(42)
        bary = rng.dirichlet([1, 1, 1], size=N)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        best_pts = np.column_stack([x, y])
        best_area = _min_area_cart(best_pts)
        
        # If time remains, try a few NM restarts from DE result or random
        if not _is_time_up():
            from scipy.optimize import minimize
            
            def objective_nm(pts_flat):
                pts = pts_flat.reshape(N, 2)
                x = pts[:, 0]
                y = pts[:, 1]
                d1 = np.maximum(0, -y)
                d2 = np.maximum(0, y - SQRT3 * x)
                d3 = np.maximum(0, y + SQRT3 * x - SQRT3)
                penalty = 100.0 * (np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2))
                area = _min_area_cart(pts)
                return -area + penalty

            # Try polishing the DE result if it exists
            if res is not None:
                x0 = res.x
                try:
                    res_nm = minimize(objective_nm, x0, method='Nelder-Mead', 
                                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                    cand = res_nm.x.reshape(N, 2)
                    if np.all(cand[:, 1] >= -1e-8) and np.all(cand[:, 1] <= SQRT3 * cand[:, 0] + 1e-8) and np.all(cand[:, 1] <= -SQRT3 * cand[:, 0] + SQRT3 + 1e-8):
                        area = _min_area_cart(cand)
                        if area > best_area:
                            best_area = area
                            best_pts = cand.copy()
                except:
                    pass

            # Try a few random restarts if time permits
            for i in range(5):
                if _is_time_up():
                    break
                bary = rng.dirichlet([1, 1, 1], size=N)
                x = bary[:, 1] + 0.5 * bary[:, 2]
                y = H * bary[:, 2]
                x0 = np.column_stack([x, y]).flatten()
                
                try:
                    res_nm = minimize(objective_nm, x0, method='Nelder-Mead', 
                                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                    cand = res_nm.x.reshape(N, 2)
                    if np.all(cand[:, 1] >= -1e-8) and np.all(cand[:, 1] <= SQRT3 * cand[:, 0] + 1e-8) and np.all(cand[:, 1] <= -SQRT3 * cand[:, 0] + SQRT3 + 1e-8):
                        area = _min_area_cart(cand)
                        if area > best_area:
                            best_area = area
                            best_pts = cand.copy()
                except:
                    pass

    # Final check and clamp
    if best_pts is None:
        # Absolute fallback
        bary = np.ones((N, 3)) / 3.0
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        best_pts = np.column_stack([x, y])

    # Clamp to triangle just in case
    x = best_pts[:, 0]
    y = best_pts[:, 1]
    y = np.maximum(0, y)
    y = np.minimum(y, SQRT3 * x)
    y = np.minimum(y, -SQRT3 * x + SQRT3)
    best_pts[:, 1] = y
    
    return best_pts
