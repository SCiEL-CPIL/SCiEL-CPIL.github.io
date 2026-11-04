"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    """Compute area of convex hull."""
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    areas = _tri_areas(pts)
    min_area = np.min(areas)
    return float(min_area / ha)

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Returns the ratio. Higher is better.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    ha = _hull_area(pts)
    if ha < 1e-12:
        return -1e9
    
    areas = _tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / ha

def _neg_soft_min_ratio(pts, temperature):
    """Negative of soft min ratio for minimization."""
    val = _soft_min_ratio(pts, temperature)
    if val < -1e8:
        return 1e8
    return -val

def _neg_exact_score(pts):
    """Negative of exact score for minimization."""
    return -_score(pts)

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _generate_initial_points(seed):
    """Generate diverse initial points."""
    rng = np.random.RandomState(seed)
    # Regular polygon
    pts1 = _regular_polygon(N, radius=1.0)
    # Jittered regular polygon
    pts2 = pts1 + rng.normal(0, 0.1, (N, 2))
    # Random points in disk
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2*np.pi, N)
    pts3 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    return [pts1, pts2, pts3]

def _local_refinement(pts, time_budget, rng):
    """
    Refine a point set using L-BFGS-B on soft-min and Nelder-Mead on exact score.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Stage 1: L-BFGS-B with soft-min surrogate
    # Use a decreasing temperature schedule
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 5:
            break
        
        def obj(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp)
        
        try:
            result = minimize(
                obj, best_pts.flatten(), method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = result.x.reshape(N, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.copy()
        except Exception:
            pass
        
        if time.time() - start_time > time_budget - 5:
            break

    # Stage 2: Nelder-Mead on exact score
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            return _neg_exact_score(x.reshape(N, 2))
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand = result.x.reshape(N, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.copy()
        except Exception:
            pass

    return best_pts, best_score

def _differential_evolution_cartesian(start_pts, time_budget, rng):
    """
    Global search in Cartesian space using Differential Evolution.
    Uses soft-min for evaluation during evolution, exact score for tracking best.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Population size
    pop_size = 30
    
    # Initialize population
    pop = np.zeros((pop_size, N, 2))
    pop_scores = np.zeros(pop_size)
    
    # First individual is the start point
    pop[0] = start_pts
    pop_scores[0] = _score(start_pts)
    
    # Rest are random perturbations or random points
    for i in range(1, pop_size):
        if rng.random() < 0.5:
            # Perturb start point
            pop[i] = start_pts + rng.normal(0, 0.2, (N, 2))
        else:
            # Random points in disk
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2*np.pi, N)
            pop[i] = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        pop_scores[i] = _score(pop[i])
        
        if time.time() > deadline - 10:
            break
    
    best_idx = np.argmax(pop_scores)
    best_pts = pop[best_idx].copy()
    best_score = pop_scores[best_idx]
    
    # DE Parameters
    F = 0.8
    CR = 0.9
    
    # Temperature for soft-min during DE
    # Start with higher temp for exploration, decrease for exploitation
    current_temp = 0.1
    
    while time.time() < deadline - 15:
        # Decrease temperature slowly
        current_temp *= 0.95
        if current_temp < 0.005:
            current_temp = 0.005
            
        new_pop = np.zeros_like(pop)
        new_scores = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select 3 distinct random individuals
            idxs = rng.choice(pop_size, 3, replace=False)
            a_idx, b_idx, c_idx = idxs
            
            # Create mutant vector
            v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            
            # Clip to reasonable bounds (e.g., [-2, 2] to keep hull area reasonable)
            v = np.clip(v, -2.0, 2.0)
            
            # Crossover
            trial = np.zeros_like(v)
            j_rand = rng.randint(0, 2*N)
            for j in range(2*N):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            
            trial_pts = trial.reshape(N, 2)
            
            # Evaluate using soft-min for smoothness, but track exact score for best
            # To save time, we use soft-min for the "fitness" comparison in DE,
            # but we need to be careful. Actually, for DE, using the exact score
            # is better if it's fast enough. With N=13, exact score is fast.
            # Let's use exact score for correctness, as vectorization makes it fast.
            
            trial_score = _score(trial_pts)
            
            # If trial is better than current, replace
            if trial_score >= pop_scores[i]:
                new_pop[i] = trial_pts
                new_scores[i] = trial_score
            else:
                new_pop[i] = pop[i]
                new_scores[i] = pop_scores[i]
                
            # Update global best
            if trial_score > best_score:
                best_score = trial_score
                best_pts = trial_pts.copy()
                
        pop = new_pop
        pop_scores = new_scores
        
        # Check time
        if time.time() > deadline - 15:
            break

    return best_pts, best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate initial diverse starts
    starts = _generate_initial_points(42)
    
    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first)
    scored_starts = []
    for start in starts:
        try:
            s = _score(start)
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Strategy:
    # 1. Run DE from the best start with a significant time budget.
    # 2. Run DE from the second best start with a smaller budget.
    # 3. Refine the best found so far.
    
    n_starts = len(started_starts) if 'started_starts' in locals() else len(scored_starts)
    
    # Allocate time
    # 60% to DE on best start
    # 20% to DE on second best start
    # 20% to refinement
    
    time_de1 = (total_budget - 5) * 0.6
    time_de2 = (total_budget - 5) * 0.2
    time_refine = (total_budget - 5) * 0.2
    
    # Run DE on best start
    if time.time() - start_time < total_budget - 5:
        try:
            de_pts, de_score = _differential_evolution_cartesian(
                scored_starts[0][1], time_de1, rng
            )
            if de_score > best_ratio:
                best_ratio = de_score
                best_points = de_pts.copy()
        except Exception:
            pass
    
    # Run DE on second best start if time permits
    if len(scored_starts) > 1 and time.time() - start_time < total_budget - 5:
        try:
            de_pts2, de_score2 = _differential_evolution_cartesian(
                scored_starts[1][1], time_de2, rng
            )
            if de_score2 > best_ratio:
                best_ratio = de_score2
                best_points = de_pts2.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        remaining_time = total_budget - (time.time() - start_time) - 1.0
        if remaining_time > 5.0:
            try:
                final_pts, final_score = _local_refinement(
                    best_points, remaining_time, rng
                )
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
