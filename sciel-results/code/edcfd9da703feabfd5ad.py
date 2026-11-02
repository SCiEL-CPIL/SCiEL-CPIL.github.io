"""Optimize 14 points in 3D to minimize squared max/min pairwise distance ratio."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# Deadline guard: finish well before 150s wall-clock
_DEADLINE = 140.0


def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances (N x N)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.einsum('ijk,ijk->ij', diff, diff)


def _score(P):
    """Compute (max/min)^2 of pairwise distances; return inf if degenerate."""
    D2 = _pairwise_dists_sq(P)
    np.fill_diagonal(D2, np.inf)
    dmin = np.min(D2)
    dmax = np.max(D2)
    if dmin <= 1e-30:
        return np.inf
    return (dmax / dmin) ** 2


def _smooth_objective(x, t, soft_k):
    """
    Smooth surrogate: minimize t subject to soft-max constraint.
    x: flattened point positions (N*D,)
    t: auxiliary variable for max distance^2
    We use log-sum-exp style soft-max for the max, and a penalty for min.
    Actually, let's use a different smooth formulation:
    minimize  (soft_max_d2 - soft_min_d2)^2 / soft_min_d2^2
    via log-sum-exp for soft_max and log-sum-exp on negative for soft_min.
    """
    P = x.reshape(N, D)
    D2 = _pairwise_dists_sq(P)
    np.fill_diagonal(D2, np.inf)
    # soft max via log-sum-exp
    m = np.max(D2)
    soft_max = m + np.log(np.sum(np.exp(D2 - m))) / (N * (N - 1))
    # soft min via log-sum-exp on negative
    mn = np.min(D2)
    soft_min = mn - np.log(np.sum(np.exp(mn - D2))) / (N * (N - 1))
    if soft_min <= 1e-30:
        return 1e10
    ratio = soft_max / soft_min
    return ratio * ratio + 1e-6 * t * t  # small regularization on t


def _initial_tetrahedron_midpoints():
    """
    Create 14 points: 4 vertices of a regular tetrahedron + 6 edge midpoints + 4 face centroids.
    Actually, tetrahedron has 4 vertices, 6 edges, 4 faces.
    4 + 6 + 4 = 14.
    """
    # Regular tetrahedron vertices (centered at origin, side length 2)
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    # Scale to have unit circumradius or just use as-is
    # Vertices
    points = [v[0], v[1], v[2], v[3]]
    # Edge midpoints (6 edges)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        points.append((v[i] + v[j]) / 2.0)
    # Face centroids (4 faces)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for face in faces:
        centroid = (v[face[0]] + v[face[1]] + v[face[2]]) / 3.0
        points.append(centroid)
    
    P = np.array(points, dtype=float)
    # Center at origin
    P -= np.mean(P, axis=0)
    return P


def _initial_random(seed=42):
    """Random initial points on a sphere."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((N, D))
    # Normalize to unit sphere
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    return P


def _initial_grid():
    """Grid-based initial points."""
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    P = np.array(grid, dtype=float)
    P -= np.mean(P, axis=0)
    return P


def _ga_optimize(P_init, time_limit=30.0):
    """
    Genetic algorithm to refine point positions.
    Minimizes the squared ratio with a penalty for distinctness.
    """
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    pop_size = 40
    elite_size = 5
    mutation_rate = 0.15
    crossover_rate = 0.8
    
    # Initialize population
    population = [P_init.copy()]
    for _ in range(pop_size - 1):
        P = P_init + rng.standard_normal((N, D)) * 0.3
        population.append(P)
    
    def fitness(P):
        D2 = _pairwise_dists_sq(P)
        np.fill_diagonal(D2, np.inf)
        dmin = np.min(D2)
        dmax = np.max(D2)
        if dmin <= 1e-20:
            return 1e10
        score = (dmax / dmin) ** 2
        # Penalty for very close points
        close_pairs = np.sum(D2 < 1e-6)
        return score + close_pairs * 100.0
    
    best_P = None
    best_score = np.inf
    
    generation = 0
    while time.time() - start_time < time_limit:
        # Evaluate fitness
        scores = [fitness(P) for P in population]
        
        # Track best
        idx = np.argmin(scores)
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_P = population[idx].copy()
        
        # Sort by fitness
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism
        new_population = population[:elite_size]
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if rng.random() < crossover_rate and len(population) >= 2:
                # Tournament selection
                i = rng.integers(0, min(5, len(population)))
                j = rng.integers(0, min(5, len(population)))
                parent1 = population[i]
                parent2 = population[j]
                # Blend crossover
                alpha = rng.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = population[rng.integers(0, min(5, len(population)))].copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                child += rng.standard_normal((N, D)) * 0.1
            else:
                child += rng.standard_normal((N, D)) * 0.02
            
            new_population.append(child)
        
        population = new_population
        generation += 1
    
    return best_P, best_score


def _polish(P_init, time_limit=20.0):
    """
    High-precision polish using scipy minimize on the smooth objective.
    """
    start_time = time.time()
    x0 = P_init.flatten()
    
    def objective(x):
        P = x.reshape(N, D)
        D2 = _pairwise_dists_sq(P)
        np.fill_diagonal(D2, np.inf)
        m = np.max(D2)
        soft_max = m + np.log(np.sum(np.exp(D2 - m))) / (N * (N - 1))
        mn = np.min(D2)
        soft_min = mn - np.log(np.sum(np.exp(mn - D2))) / (N * (N - 1))
        if soft_min <= 1e-30:
            return 1e10
        ratio = soft_max / soft_min
        return ratio * ratio
    
    best_P = P_init.copy()
    best_score = _score(P_init)
    
    # Try multiple starting points for polish
    for seed_offset in range(3):
        if time.time() - start_time > time_limit:
            break
        
        if seed_offset == 0:
            x_start = P_init.flatten()
        else:
            rng = np.random.default_rng(100 + seed_offset)
            x_start = P_init.flatten() + rng.standard_normal(N * D) * 0.05
        
        try:
            result = minimize(
                objective, x_start, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            P_candidate = result.x.reshape(N, D)
            score_candidate = _score(P_candidate)
            
            if score_candidate < best_score:
                best_score = score_candidate
                best_P = P_candidate
        except Exception:
            continue
    
    return best_P, best_score


def construct():
    start_time = time.time()
    
    # Strategy 1: Start from tetrahedron + midpoints + centroids
    P_tet = _initial_tetrahedron_midpoints()
    score_tet = _score(P_tet)
    
    # Strategy 2: Random initial
    P_rand = _initial_random(seed=42)
    score_rand = _score(P_rand)
    
    # Strategy 3: Grid
    P_grid = _initial_grid()
    score_grid = _score(P_grid)
    
    # Choose best initial
    candidates = [
        (P_tet, score_tet),
        (P_rand, score_rand),
        (P_grid, score_grid),
    ]
    candidates.sort(key=lambda x: x[1])
    P_init, _ = candidates[0]
    
    # Genetic algorithm refinement
    P_ga, score_ga = _ga_optimize(P_init, time_limit=60.0)
    
    # Polish with scipy
    P_polished, score_polished = _polish(P_ga, time_limit=20.0)
    
    # Also try polishing other candidates if time permits
    best_P = P_polished
    best_score = score_polished
    
    if time.time() - start_time < _DEADLINE - 5:
        for P_cand, _ in candidates[1:]:
            if time.time() - start_time > _DEADLINE - 5:
                break
            P_p, s_p = _polish(P_cand, time_limit=10.0)
            if s_p < best_score:
                best_score = s_p
                best_P = P_p
    
    # Final check
    final_score = _score(best_P)
    if final_score < best_score:
        best_score = final_score
    
    return best_P
