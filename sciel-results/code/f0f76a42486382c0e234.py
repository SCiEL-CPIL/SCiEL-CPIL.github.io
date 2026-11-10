import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill
_params_cache = None


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """
    Simulate ACO and return the highest total prize collected by any ant in any iteration.
    Optimized for speed: minimal overhead in the inner loop.
    """
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()
    
    # Pre-allocate visited array to avoid reallocation? 
    # Actually, creating a new bool array each ant is fast enough in numpy.
    
    global_best = 0.0

    for it in range(num_iters):
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                # Vectorized feasibility check
                # dist[current, :] is a view, dist_to_home is precomputed
                feasible = (~visited) & ((dist[current, :] + dist_to_home) <= remaining)
                feasible[0] = False

                if not np.any(feasible):
                    break

                candidates = np.where(feasible)[0]
                weights = pheromone[current, candidates] * heuristic[current, candidates]

                total_w = weights.sum()
                if total_w <= 0:
                    break

                probs = weights / total_w
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best


def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix.
    params: (a, b, c, d)
    h[i,j] = prize[j]^a / (d_ij^b * (d_j0+eps)^c) * (budget_norm)^d
    
    Note: I removed 'e' from the previous version and combined it into 'a' to reduce dimensionality,
    or kept the structure but simplified. Let's stick to 4 params for better search efficiency.
    a: prize exponent
    b: inverse distance exponent
    c: inverse distance-to-home exponent
    d: budget factor exponent
    """
    n = len(prizes)
    a, b, c, d = params
    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :]  # (1, n)
    
    # budget: max_len - d_ij - d_j0 (how much length remains after visiting j)
    budget = np.maximum(0.0, max_len - dist - d_j0_row)  # (n, n)
    budget_norm = budget / max_len  # normalized to [0, 1]
    
    # Core: prize^power / (d_ij^b * d_j0^c)
    h = (prize ** a) / (d_ij ** b * d_j0_row ** c)
    
    # Budget factor: reward nodes that leave more budget
    # Adding a small epsilon to budget_norm to avoid 0^d issues if d>0, though max(0,...) handles it.
    # If budget is 0, h becomes 0. This is a strong penalty.
    h = h * (budget_norm + 1e-9) ** d
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _evaluate_params(params, instances, eval_rng):
    """Evaluate a set of parameters on all instances."""
    total_score = 0.0
    for prizes, dist, max_len in instances:
        n = len(prizes)
        heuristic = _make_heuristic(prizes, dist, max_len, params)
        score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
        total_score += score
    return total_score / len(instances)


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    # Generate representative instances
    # Use a fixed set of seeds for reproducibility and speed
    n_instances = 10
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed * 13 + 7)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = (1.0, 1.0, 0.5, 0.5)
    best_score = -1.0
    eval_rng = np.random.RandomState(123)

    # Phase 1: Coarse Grid Search
    # a: prize exponent [0.5, 1.0, 1.5, 2.0, 2.5]
    # b: inv dist exponent [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # c: inv dist-to-home exponent [0.0, 0.5, 1.0, 1.5]
    # d: budget exponent [0.0, 0.5, 1.0, 1.5, 2.0]
    
    a_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    b_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    c_vals = [0.0, 0.5, 1.0, 1.5]
    d_vals = [0.0, 0.5, 1.0, 1.5, 2.0]

    candidates = []
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                for d in d_vals:
                    candidates.append((a, b, c, d))
    
    # Shuffle candidates to avoid bias if we have to stop early? 
    # No, better to check time before each eval.
    
    for params in candidates:
        if time.time() > _DEADLINE - 20.0:
            break
            
        score = _evaluate_params(params, instances, eval_rng)
        if score > best_score:
            best_score = score
            best_params = params

    # Phase 2: Local Refinement (Random Search around best)
    if time.time() < _DEADLINE - 10.0:
        base_params = np.array(best_params)
        # Perform random perturbations
        for _ in range(100): # Max 100 steps, but limited by time
            if time.time() > _DEADLINE - 5.0:
                break
            
            # Gaussian noise
            noise = np.random.randn(4) * 0.2
            new_params = base_params + noise
            # Ensure params are within reasonable bounds
            new_params = np.clip(new_params, [0.1, 0.1, 0.0, 0.0], [3.0, 3.0, 2.0, 2.0])
            
            score = _evaluate_params(tuple(new_params), instances, eval_rng)
            if score > best_score:
                best_score = score
                best_params = tuple(new_params)
                base_params = new_params # Move base to new best
            else:
                # If not better, we can just keep searching around the global best 
                # or revert. Let's just keep searching the neighborhood of the current global best.
                pass

    _params_cache = best_params
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    return heuristic
