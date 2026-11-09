import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def get_score(points):
    """
    Computes the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0

    # Vectorized triangle area computation
    idx = np.array(list(itertools.combinations(range(N), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    N = 13
    DIM = 2 * N  # 26 dimensions
    
    # Initial population: mix of regular polygons with random rotations/scales
    # and some random convex-ish points
    pop_size = 50
    population = np.zeros((pop_size, DIM))
    scores = np.zeros(pop_size)
    
    # Generate initial population
    for i in range(pop_size):
        if i < pop_size // 2:
            # Regular polygon variants
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.uniform(0, 2 * np.pi)
            scale = np.random.uniform(0.5, 1.5)
            pts = np.column_stack((scale * np.cos(angles), scale * np.sin(angles)))
        else:
            # Random points, ensure convexity by sorting by angle around centroid
            pts = np.random.uniform(-1, 1, (N, 2))
            centroid = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            sorted_idx = np.argsort(angles)
            pts = pts[sorted_idx]
        
        population[i] = pts.flatten()
        scores[i] = get_score(pts)
    
    best_score = np.max(scores)
    best_points = population[np.argmax(scores)].reshape(N, 2).copy()
    
    # Differential Evolution Parameters
    F = 0.5  # Scale factor
    CR = 0.9  # Crossover rate
    # We will adapt F and CR slightly or keep them fixed. Fixed is simpler and often works well.
    
    # Time budget for DE
    # Each generation involves pop_size evaluations.
    # get_score is fast (vectorized), but ConvexHull is the bottleneck.
    # Let's estimate time per evaluation.
    
    # We run DE until time runs out.
    while time.time() < deadline - 5.0:
        # Generate trial population
        trial_population = np.zeros_like(population)
        trial_scores = np.zeros_like(pop_size)
        
        for i in range(pop_size):
            # Select three distinct individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            
            # Differential variation
            mutant = population[a] + F * (population[b] - population[c])
            
            # Binary crossover
            # Generate random indices for crossover
            cross_mask = np.random.rand(DIM) < CR
            # Ensure at least one gene is mutated
            if not np.any(cross_mask):
                cross_mask[np.random.randint(DIM)] = True
            
            trial = population[i].copy()
            trial[cross_mask] = mutant[cross_mask]
            
            trial_population[i] = trial
            trial_pts = trial.reshape(N, 2)
            trial_scores[i] = get_score(trial_pts)
            
            # Selection
            if trial_scores[i] > scores[i]:
                scores[i] = trial_scores[i]
                population[i] = trial
                if trial_scores[i] > best_score:
                    best_score = trial_scores[i]
                    best_points = trial_pts.copy()
        
        # Adaptive cooling of F and CR to encourage convergence later?
        # Or just keep it simple.
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final local refinement using Nelder-Mead on the best point found
    if time.time() < deadline - 2.0 and best_points is not None:
        from scipy.optimize import minimize
        
        def objective_local(params):
            points = params.reshape(N, 2)
            try:
                hull = ConvexHull(points)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e10
                idx = np.array(list(itertools.combinations(range(N), 3)))
                p1 = points[idx[:, 0]]
                p2 = points[idx[:, 1]]
                p3 = points[idx[:, 2]]
                cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
                        (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
                areas = 0.5 * np.abs(cross)
                min_area = np.min(areas)
                return -min_area / hull_area
            except:
                return 1e10

        try:
            res = minimize(objective_local, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < -1e9:
                return best_points
            
            final_score = -res.fun
            pts_final = res.x.reshape(N, 2)
            exact_final_score = get_score(pts_final)
            
            if exact_final_score > best_score:
                best_score = exact_final_score
                best_points = pts_final.copy()
        except Exception:
            pass

    return best_points
