"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(TRIPLES)

def _clamp_bary(bary):
    u = np.clip(bary[:, 0], 0.0, 1.0)
    v = np.clip(bary[:, 1], 0.0, 1.0)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * (np.sqrt(3) / 2)
    return np.column_stack((x, y))

def _all_dets(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    i_idx = TRIPLES[:, 0]
    j_idx = TRIPLES[:, 1]
    k_idx = TRIPLES[:, 2]
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det

def _min_det(bary):
    return np.min(np.abs(_all_dets(bary)))

def _softmin_obj(x, beta=80.0):
    bary = _clamp_bary(x.reshape(N, 2))
    dets = _all_dets(bary)
    areas = np.abs(dets)
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _hard_obj(x):
    bary = _clamp_bary(x.reshape(N, 2))
    return -_min_det(bary)

def _fps_init(rng):
    n_grid = 40
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    selected = [np.array([1.0/3, 1.0/3])]
    for _ in range(N - 1):
        dists = np.array([np.min(np.linalg.norm(c - selected, axis=1)) for c in candidates])
        selected.append(candidates[np.argmax(dists)])
    return np.array(selected)

def _random_init(rng):
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0 - u)
            bary[i] = [u, v]
            break
    return bary

def _symmetric_init(rng):
    # 3 vertices + 3 edge midpoints + centroid + 3 more
    bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
        [0.2, 0.2], [0.6, 0.2], [0.2, 0.6], [0.4, 0.4]
    ])
    jitter = rng.normal(0, 0.03, size=bary.shape)
    return _clamp_bary(bary + jitter)

def _differential_evolution(bary0, pop_size=40, max_gen=200, F=0.8, CR=0.7, beta=80.0):
    """DE on soft-min objective in barycentric space."""
    rng = np.random.RandomState(42)
    dim = 2 * N
    pop = np.zeros((pop_size, dim))
    pop[0] = bary0.flatten()
    for i in range(1, pop_size):
        pop[i] = _random_init(rng).flatten()
    # Clamp all
    for i in range(pop_size):
        pop[i] = _clamp_bary(pop[i].reshape(N, 2)).flatten()
    
    best_bary = bary0.copy()
    best_score = _min_det(bary0)
    
    # Evaluate initial population
    scores = np.array([_min_det(pop[i].reshape(N, 2)) for i in range(pop_size)])
    for i in range(pop_size):
        if scores[i] > best_score:
            best_score = scores[i]
            best_bary = pop[i].reshape(N, 2).copy()
    
    for gen in range(max_gen):
        new_pop = pop.copy()
        new_scores = scores.copy()
        for i in range(pop_size):
            # Select three distinct random indices
            a, b, c = rng.choice(pop_size, 3, replace=False)
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            # Crossover
            trial = np.zeros(dim)
            j_rand = rng.randint(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            # Clamp
            trial_bary = _clamp_bary(trial.reshape(N, 2))
            trial_score = _min_det(trial_bary)
            if trial_score >= scores[i]:
                new_pop[i] = trial_bary.flatten()
                new_scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_bary = trial_bary.copy()
        pop = new_pop
        scores = new_scores
    return best_bary, best_score

def _simulated_annealing(bary0, max_iter=5000, T0=0.05, T_end=0.001, beta=80.0):
    """SA on soft-min objective."""
    rng = np.random.RandomState(77)
    dim = 2 * N
    x = bary0.flatten()
    current_score = _min_det(bary0)
    best_bary = bary0.copy()
    best_score = current_score
    
    for it in range(max_iter):
        T = T0 * (T_end / T0) ** (it / max_iter)
        scale = 0.02 + 0.03 * (T / T0)
        pert = rng.normal(0, scale, size=dim)
        x_trial = _clamp_bary((x + pert).reshape(N, 2)).flatten()
        trial_score = _min_det(x_trial.reshape(N, 2))
        delta = trial_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            x = x_trial
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_bary = x_trial.reshape(N, 2).copy()
    return best_bary, best_score

def _local_refine(bary0, method='slsqp', maxiter=200):
    """Final local refinement."""
    try:
        x0 = bary0.flatten()
        if method == 'slsqp':
            bounds = [(0, 1)] * (2 * N)
            constraints = []
            for i in range(N):
                def cons(x, i=i):
                    return 1.0 - (x[2*i] + x[2*i+1])
                constraints.append({'type': 'ineq', 'fun': cons})
            res = minimize(_hard_obj, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-14})
        elif method == 'nm':
            res = minimize(_hard_obj, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True})
        else:
            res = minimize(_hard_obj, x0, method='L-BFGS-B',
                          bounds=[(0,1)]*(2*N), options={'maxiter': maxiter, 'ftol': 1e-15})
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_bary = None
    best_score = -np.inf
    
    # Generate diverse starting points
    rng = np.random.RandomState(12345)
    starts = []
    starts.append(_fps_init(rng))
    starts.append(_symmetric_init(rng))
    for _ in range(5):
        starts.append(_random_init(rng))
    
    # Phase 1: DE from each start (limited gens to stay within budget)
    for idx, bary0 in enumerate(starts):
        if time_left() < 10.0:
            break
        bary_de, score_de = _differential_evolution(bary0, pop_size=30, max_gen=80, F=0.8, CR=0.7)
        if score_de > best_score:
            best_score = score_de
            best_bary = bary_de.copy()
    
    # Phase 2: SA refinement from best
    if time_left() > 15.0:
        bary_sa, score_sa = _simulated_annealing(best_bary, max_iter=3000, T0=0.03, T_end=0.0005)
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
    
    # Phase 3: Local refinement
    if time_left() > 10.0:
        bary_opt = _local_refine(best_bary, method='slsqp', maxiter=200)
        score_opt = _min_det(bary_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    # Phase 4: More SA from slightly perturbed best
    if time_left() > 15.0:
        for _ in range(3):
            if time_left() < 8.0:
                break
            pert = rng.normal(0, 0.02, size=(N, 2))
            bary_start = _clamp_bary(best_bary + pert)
            bary_sa, score_sa = _simulated_annealing(bary_start, max_iter=2000, T0=0.02, T_end=0.0005)
            if score_sa > best_score:
                best_score = score_sa
                best_bary = bary_sa.copy()
    
    # Phase 5: Final polish
    if time_left() > 8.0:
        bary_opt = _local_refine(best_bary, method='nm', maxiter=200)
        score_opt = _min_det(bary_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    if time_left() > 5.0:
        bary_opt = _local_refine(best_bary, method='slsqp', maxiter=300)
        score_opt = _min_det(bary_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    if best_bary is None:
        best_bary = _fps_init(rng)
    
    return _bary_to_cart(best_bary)
