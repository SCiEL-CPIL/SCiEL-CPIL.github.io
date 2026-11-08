import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 13
# Precompute all triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRI = len(_TRI_IDX)

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if len(points) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def neg_score_softmin(points_flat, beta):
    """
    Smoothed objective using log-sum-exp soft-min of triangle areas / hull area.
    Minimizes - (softmin(areas) / hull_area)
    """
    try:
        points = points_flat.reshape(N, 2)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6
        except Exception:
            return 1e6
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Soft min calculation
        # soft_min = -1/beta * log(sum(exp(-beta * areas)))
        # To avoid overflow, shift by max area
        max_area = np.max(areas)
        shifted = areas - max_area
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = max_area - (1.0 / beta) * log_sum
        
        return -soft_min_val / hull_area
    except Exception:
        return 1e6

def generate_seeds():
    """
    Generate a variety of structured initial points.
    """
    seeds = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    seeds.append(init_reg)
    
    # 2. Noisy Regular 13-gon
    for _ in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        seeds.append(init_reg + noise)
        
    # 3. Two rings: 7 outer, 6 inner
    # Vary the inner radius and rotation
    for r_in in [0.3, 0.5, 0.7, 0.8]:
        for rot in [0, np.pi/6, np.pi/3]:
            angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_6 = np.linspace(rot, 2 * np.pi + rot, 6, endpoint=False)
            p7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
            p6 = np.column_stack((r_in * np.cos(angles_6), r_in * np.sin(angles_6)))
            seeds.append(np.vstack([p7, p6]))
            
    # 4. Three rings: 5 outer, 4 mid, 4 inner
    for r_mid, r_in in [(0.6, 0.3), (0.7, 0.5), (0.8, 0.6)]:
        angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        angles_4m = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        angles_4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
        p5 = np.column_stack((np.cos(angles_5), np.sin(angles_5)))
        p4m = np.column_stack((r_mid * np.cos(angles_4m), r_mid * np.sin(angles_4m)))
        p4i = np.column_stack((r_in * np.cos(angles_4i), r_in * np.sin(angles_4i)))
        seeds.append(np.vstack([p5, p4m, p4i]))
        
    # 5. "Star" shape: Alternating radii
    # 13 points. 7 outer, 6 inner alternating? No, 13 is odd.
    # Let's try 1 outer, 2 inner, 1 outer... 
    # Or just perturb the regular polygon radially.
    for _ in range(5):
        radii = 1.0 + rng.normal(0, 0.2, N)
        radii = np.clip(radii, 0.5, 1.5)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        p = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        seeds.append(p)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    def update_best(points):
        nonlocal best_score, best_points
        s = score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Initial seeds
    seeds = generate_seeds()
    for seed in seeds:
        update_best(seed)
        
    if best_points is None:
        best_points = seeds[0]

    bounds = [(-2.0, 2.0)] * (N * 2)

    # Strategy 1: Global Differential Evolution
    # Use a soft-min with beta=50 to keep landscape smooth for DE
    beta_de = 50.0
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                neg_score_softmin,
                bounds=bounds,
                args=(beta_de,),
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                callback=None # We'll check time manually if needed, but DE handles maxiter
            )
            
            if de_result.fun < 1e5:
                de_points = de_result.x.reshape(N, 2)
                update_best(de_points)
    except Exception:
        pass
    
    # Strategy 2: Local Optimization (L-BFGS-B) with Annealing
    # Polish the best points found so far and the initial seeds
    
    candidates = list(seeds)
    if best_points is not None:
        candidates.append(best_points)
        
    # Add noisy versions of the best points
    rng = np.random.default_rng(42)
    if best_points is not None:
        for _ in range(5):
            noise = rng.normal(0, 0.05, (N, 2))
            candidates.append(best_points + noise)

    # Annealing schedule for beta
    betas = [50.0, 200.0, 1000.0]
    
    for start in candidates:
        if time.time() > deadline - 20:
            break
            
        current_points = start.flatten()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    neg_score_softmin,
                    current_points,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.fun < 1e5:
                    candidate_points = res.x.reshape(N, 2)
                    update_best(candidate_points)
                    current_points = res.x
            except Exception:
                pass

    # Strategy 3: Final Polish with True Objective
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_true = minimize(
                neg_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res_true.fun < 0:
                update_best(res_true.x.reshape(N, 2))
        except Exception:
            pass

    # Strategy 4: Second DE pass around best point if time allows
    if best_points is not None and time.time() < deadline - 30:
        try:
            center = best_points.mean(axis=0)
            spread = np.abs(best_points - center).max()
            b = max(spread * 1.2, 0.5)
            
            bounds_tight_flat = []
            for i in range(N):
                bounds_tight_flat.append((best_points[i,0]-b, best_points[i,0]+b))
                bounds_tight_flat.append((best_points[i,1]-b, best_points[i,1]+b))
            
            de2 = differential_evolution(
                neg_score_softmin,
                bounds=bounds_tight_flat,
                args=(200.0,), # Higher beta for local refinement
                seed=123,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                mutation=(0.2, 0.8),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            
            if de2.fun < 1e5:
                de2_points = de2.x.reshape(N, 2)
                # Polish the result of the second DE
                res_p = minimize(
                    neg_score_softmin,
                    de2_points.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_p.fun < 1e5:
                    update_best(res_p.x.reshape(N, 2))
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {score(pts)}")
