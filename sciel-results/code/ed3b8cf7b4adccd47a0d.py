import numpy as np
import time

_params_cache = None


def _simulate_aco_fast(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO with precomputed structures for speed."""
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()

    global_best = 0.0

    for it in range(num_iters):
        iter_best = 0.0

        for ant in range(num_ants):
            sub_rng = np.random.RandomState(it * 1000 + ant)
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
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
                idx = sub_rng.choice(len(candidates), p=probs)
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


def _compute_regret_matrix(dist, max_len, n):
    """
    For each node j, compute how many other nodes k can be visited after visiting j
    (i.e., d[j,k] + d[k,0] <= remaining_budget). This is a proxy for 'flexibility'.
    We use a simplified version: for each j, count nodes k where d[j,k] + d[k,0] <= max_len.
    """
    dist_to_home = dist[:, 0]
    # For each pair (j, k), check if d[j,k] + d[k,0] <= max_len
    # regret[j] = number of k such that d[j,k] + d[k,0] <= max_len
    # This tells us how many options remain if we go to j
    
    # Compute all-pairs: can[j,k] = d[j,k] + d[k,0] <= max_len
    can = (dist + dist_to_home[None, :]) <= max_len
    can[0, :] = False  # Can't "go back" to 0 in this sense
    can[:, 0] = False  # 0 is start
    
    # regret[j] = sum over k of can[j,k]
    regret_counts = can.sum(axis=1).astype(float)
    
    # Higher regret_count means more flexibility after visiting j
    # We want to AVOID nodes with low regret (dead-ends)
    # So attractiveness should be inversely related to regret being low
    # Or: reward = 1 / (1 + regret_penalty) where regret_penalty is high for low flexibility
    
    # Actually, we want to REWARD moves that preserve flexibility
    # So if going to j leaves many options, that's good
    # regret_score[j] = regret_counts[j] / max_regret (normalized)
    max_rc = regret_counts.max() if regret_counts.max() > 0 else 1.0
    flexibility = regret_counts / max_rc
    
    return flexibility, regret_counts


def _make_heuristic(prizes, dist, max_len, params):
    """
    Build heuristic matrix combining prize, distance, and regret-based flexibility.
    params: (w_prize, w_dist, w_regret, alpha, beta, gamma)
    - w_prize: weight for prize component
    - w_dist: weight for distance penalty
    - w_regret: weight for regret/flexibility component  
    - alpha: prize exponent
    - beta: distance penalty strength
    - gamma: regret penalty strength
    """
    n = len(prizes)
    w_prize, w_dist, w_regret, alpha, beta, gamma = params

    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-10       # (n, n)
    d_j0 = dist[:, 0] + 1e-10 # (n,)
    d_j0_row = d_j0[None, :]  # (1, n)

    # Prize component: higher prize = more attractive
    prize_comp = prize ** alpha
    
    # Distance component: penalize far nodes (both to reach and to return)
    total_dist = d_ij + d_j0_row
    dist_comp = np.exp(-beta * total_dist)
    
    # Regret/flexibility component: reward nodes that leave many options
    flexibility, _ = _compute_regret_matrix(dist, max_len, n)
    # flexibility[j] is high if visiting j leaves many options
    # We want to prefer moves TO nodes with high flexibility
    # So for edge i->j, use flexibility[j]
    flex_col = flexibility[None, :]  # (1, n) - same for all i
    regret_comp = flex_col ** gamma
    
    # Combine: multiplicative or additive?
    # Multiplicative: H = prize_comp * dist_comp * regret_comp
    # But this can be zero. Let's use a weighted geometric mean or product with floors.
    
    # Use product form with small epsilon to avoid zeros
    h = (prize_comp * dist_comp * regret_comp)
    
    # Ensure positive
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h


def _make_heuristic_v2(prizes, dist, max_len, params):
    """
    Alternative: additive combination with different structure.
    params: (p1, p2, p3, p4, p5)
    H[i,j] = prize[j]^p1 / (d_ij^p2 + d_j0[j]^p3 + p4)^p5 * (1 + p5 * flexibility[j])
    """
    n = len(prizes)
    p1, p2, p3, p4, p5 = params

    prize = prizes[None, :]
    d_ij = dist + 1e-10
    d_j0 = dist[:, 0] + 1e-10
    d_j0_row = d_j0[None, :]
    
    flexibility, _ = _compute_regret_matrix(dist, max_len, n)
    flex_col = 1.0 + p5 * flexibility[None, :]
    
    cost = d_ij ** p2 + d_j0_row ** p3 + p4
    h = (prize ** p1) / (cost ** p5) * flex_col
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _make_heuristic_v3(prizes, dist, max_len, params):
    """
    V3: Prize density per unit path cost, with regret adjustment.
    params: (alpha, beta, gamma, delta, epsilon)
    H[i,j] = prize[j]^alpha * exp(-beta * (d_ij + d_j0[j])) * (flexibility[j]^gamma) * (1 + delta * prize[j])
    """
    n = len(prizes)
    alpha, beta, gamma, delta, epsilon = params

    prize = prizes[None, :]
    d_ij = dist + 1e-10
    d_j0 = dist[:, 0] + 1e-10
    d_j0_row = d_j0[None, :]
    
    flexibility, _ = _compute_regret_matrix(dist, max_len, n)
    flex_col = flexibility[None, :]
    
    total_dist = d_ij + d_j0_row
    h = (prize ** alpha) * np.exp(-beta * total_dist) * (flex_col + epsilon) ** gamma * (1.0 + delta * prize)
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _make_heuristic_v4(prizes, dist, max_len, params):
    """
    V4: Hybrid - use regret to modulate the effective distance penalty.
    Nodes with high flexibility get a reduced distance penalty.
    params: (p1, p2, p3, p4, p5)
    """
    n = len(prizes)
    p1, p2, p3, p4, p5 = params

    prize = prizes[None, :]
    d_ij = dist + 1e-10
    d_j0 = dist[:, 0] + 1e-10
    d_j0_row = d_j0[None, :]
    
    flexibility, _ = _compute_regret_matrix(dist, max_len, n)
    # High flexibility -> lower effective distance penalty
    # effective_beta = p2 * (1 - p5 * flexibility[j])
    # But we need it per-edge, so:
    flex_col = flexibility[None, :]
    eff_penalty = p2 * (1.0 - p5 * flex_col)
    eff_penalty = np.maximum(eff_penalty, 0.1)  # floor
    
    cost = d_ij ** p2 + d_j0_row ** p3 + p4
    # Apply modified cost
    h = (prize ** p1) / (cost ** p3) * np.exp(-eff_penalty * (d_ij + d_j0_row))
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin

    n_instances = 6
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = None
    best_score = -1.0
    best_version = 0

    def evaluate(params, version):
        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            if version == 1:
                heuristic = _make_heuristic(prizes, dist, max_len, params)
            elif version == 2:
                heuristic = _make_heuristic_v2(prizes, dist, max_len, params)
            elif version == 3:
                heuristic = _make_heuristic_v3(prizes, dist, max_len, params)
            else:
                heuristic = _make_heuristic_v4(prizes, dist, max_len, params)
            score = _simulate_aco_fast(prizes, dist, max_len, heuristic, n, None)
            total_score += score
        return total_score / len(instances)

    # Try multiple versions with different parameter ranges
    candidates = []

    # Version 1: multiplicative (w_prize, w_dist, w_regret, alpha, beta, gamma)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [1.0, 2.0, 3.0, 4.0]:
            for gamma in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(((1.0, 1.0, 1.0, alpha, beta, gamma), 1))

    # Version 2: (p1, p2, p3, p4, p5)
    for p1 in [1.0, 1.5, 2.0]:
        for p2 in [1.0, 2.0, 3.0]:
            for p3 in [0.5, 1.0, 1.5]:
                for p4 in [0.01, 0.05, 0.1]:
                    for p5 in [0.5, 1.0, 2.0]:
                        candidates.append(((p1, p2, p3, p4, p5), 2))

    # Version 3: (alpha, beta, gamma, delta, epsilon)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [1.0, 2.0, 3.0]:
            for gamma in [0.5, 1.0, 1.5]:
                for delta in [0.0, 0.5, 1.0]:
                    candidates.append(((alpha, beta, gamma, delta, 0.01), 3))

    # Version 4: (p1, p2, p3, p4, p5)
    for p1 in [1.0, 1.5, 2.0]:
        for p2 in [0.5, 1.0, 1.5, 2.0]:
            for p3 in [0.5, 1.0, 1.5]:
                for p4 in [0.01, 0.05]:
                    for p5 in [0.3, 0.5, 0.7]:
                        candidates.append(((p1, p2, p3, p4, p5), 4))

    # Shuffle and evaluate
    import random
    random.seed(42)
    random.shuffle(candidates)

    eval_count = 0
    for params, version in candidates:
        if time.time() > deadline - 30.0:
            break
        try:
            score = evaluate(params, version)
            eval_count += 1
            if score > best_score:
                best_score = score
                best_params = params
                best_version = version
        except Exception:
            continue

    # Fine-tune around best
    if time.time() < deadline - 20.0 and best_params is not None:
        # Generate nearby candidates
        fine_candidates = []
        if best_version == 1:
            base = list(best_params)
            for i in range(len(base)):
                for delta in [-0.2, -0.1, 0.1, 0.2]:
                    new_p = base.copy()
                    new_p[i] = max(0.01, new_p[i] + delta)
                    fine_candidates.append((tuple(new_p), 1))
        elif best_version == 2:
            base = list(best_params)
            for i in range(len(base)):
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    new_p = base.copy()
                    new_p[i] = max(0.01, new_p[i] + delta)
                    fine_candidates.append((tuple(new_p), 2))
        elif best_version == 3:
            base = list(best_params)
            for i in range(len(base)):
                for delta in [-0.2, -0.1, 0.1, 0.2]:
                    new_p = base.copy()
                    new_p[i] = max(0.01, new_p[i] + delta)
                    fine_candidates.append((tuple(new_p), 3))
        else:
            base = list(best_params)
            for i in range(len(base)):
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    new_p = base.copy()
                    new_p[i] = max(0.01, new_p[i] + delta)
                    fine_candidates.append((tuple(new_p), 4))

        random.shuffle(fine_candidates)
        for params, version in fine_candidates:
            if time.time() > deadline - 10.0:
                break
            try:
                score = evaluate(params, version)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_version = version
            except Exception:
                continue

    _params_cache = (best_params, best_version)
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    params, version = _optimize_parameters()
    if version == 1:
        heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    elif version == 2:
        heuristic = _make_heuristic_v2(prizes, distance_matrix, max_len, params)
    elif version == 3:
        heuristic = _make_heuristic_v3(prizes, distance_matrix, max_len, params)
    else:
        heuristic = _make_heuristic_v4(prizes, distance_matrix, max_len, params)
    return heuristic
