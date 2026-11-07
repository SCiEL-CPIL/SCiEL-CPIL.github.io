"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _aco_eval_fast(dm, demands, capacity, H, n_ants, n_iterations, seed, time_limit):
    """
    Fast ACO simulation to estimate heuristic quality.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)

    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def _evaluate_heuristic(dm, demands, capacity, H, time_limit, n_seeds=3):
    """
    Evaluate a heuristic by running ACO simulations with multiple seeds.
    Returns the average best cost.
    """
    costs = []
    for s in range(n_seeds):
        cost = _aco_eval_fast(dm, demands, capacity, H,
                              n_ants=6, n_iterations=6, seed=s,
                              time_limit=time_limit / n_seeds)
        if cost < np.inf:
            costs.append(cost)
    if costs:
        return np.mean(costs)
    return np.inf


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a demand-aware, multi-criteria approach.
    Combines inverse distance with demand penalties to bias toward routes that
    pack heavier customers early, minimizing trips and empty-return distances.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)

    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    # Precompute Euclidean distances from coordinates
    eucl_dm = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    eucl_safe = np.where(eucl_dm > 1e-10, eucl_dm, 1e10)

    # Demand-related quantities
    # Normalize demands to [0, 1] range for weighting
    max_demand = np.max(demands) if np.max(demands) > 0 else 1.0
    norm_demands = demands / max_demand

    # Demand penalty factor: higher demand nodes get lower heuristic weight
    # This encourages ants to visit high-demand nodes when capacity is fresh
    demand_penalty = 1.0 / (1.0 + 2.0 * norm_demands)  # Range: [1/3, 1]

    best_H = None
    best_score = np.inf

    def try_candidate(H_candidate, description=""):
        nonlocal best_H, best_score, deadline, start_time
        if time.time() > deadline - 20.0:
            return

        H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0),
                              H_candidate, 1e-9)
        H_candidate = H_candidate.astype(np.float64)

        # Quick evaluation with reduced simulation
        remaining = deadline - 20.0 - time.time()
        if remaining < 1.0:
            return

        sim_time = min(1.2, max(0.4, remaining / 40))
        score = _evaluate_heuristic(dm, demands, capacity, H_candidate,
                                    time_limit=sim_time, n_seeds=3)
        if score < best_score:
            best_score = score
            best_H = H_candidate.copy()

    # ---- Strategy 1: Demand-aware inverse distance ----
    # eta[i,j] = d(i,j)^-gamma * demand_penalty[j]
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if time.time() > deadline - 25.0:
            break
        H_cand = (1.0 / (dm_safe ** gamma)) * demand_penalty[np.newaxis, :]
        try_candidate(H_cand, f"demand_inv_dist gamma={gamma}")

    # ---- Strategy 2: Demand-aware with Euclidean distance ----
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > deadline - 25.0:
            break
        H_cand = (1.0 / (eucl_safe ** gamma)) * demand_penalty[np.newaxis, :]
        try_candidate(H_cand, f"eucl_demand gamma={gamma}")

    # ---- Strategy 3: Combined distance and angle smoothness with demand ----
    # Compute angle-based smoothness: prefer edges that continue in similar direction
    # For each node i, compute the angle between incoming and outgoing edges
    # This is approximated by looking at local geometry

    # Strategy 4: Multi-criteria with different demand penalty strengths
    for penalty_strength in [1.0, 2.0, 3.0, 5.0]:
        for gamma in [1.0, 1.5, 2.0]:
            if time.time() > deadline - 25.0:
                break
            dp = 1.0 / (1.0 + penalty_strength * norm_demands)
            H_cand = (1.0 / (dm_safe ** gamma)) * dp[np.newaxis, :]
            try_candidate(H_cand, f"multi_crit ps={penalty_strength} g={gamma}")

    # ---- Strategy 5: Source-dependent demand penalty ----
    # Penalize based on source node's remaining capacity proxy
    # Approximate: nodes with higher demand as source have less "remaining" capacity
    src_penalty = 1.0 / (1.0 + 1.5 * norm_demands)  # Lower for high-demand sources
    for gamma in [1.0, 1.5, 2.0]:
        if time.time() > deadline - 25.0:
            break
        H_cand = (1.0 / (dm_safe ** gamma)) * src_penalty[:, np.newaxis] * demand_penalty[np.newaxis, :]
        try_candidate(H_cand, f"src_dst_demand g={gamma}")

    # ---- Strategy 6: Exponential decay with demand ----
    for alpha in [1.0, 2.0, 3.0]:
        for gamma in [1.0, 1.5]:
            if time.time() > deadline - 25.0:
                break
            H_cand = np.exp(-alpha * dm_safe) * demand_penalty[np.newaxis, :]
            try_candidate(H_cand, f"exp_demand a={alpha} g={gamma}")

    # ---- Strategy 7: Fine-tune around best found ----
    if best_H is not None and time.time() < deadline - 40.0:
        # Do a local search around the best gamma for demand-aware inverse distance
        for gamma in np.linspace(0.5, 3.5, 30):
            if time.time() > deadline - 30.0:
                break
            H_cand = (1.0 / (dm_safe ** gamma)) * demand_penalty[np.newaxis, :]
            try_candidate(H_cand, f"fine_tune gamma={gamma:.3f}")

    # ---- Strategy 8: Even finer local search if time permits ----
    if best_H is not None and time.time() < deadline - 30.0:
        for gamma in np.linspace(0.8, 2.5, 40):
            if time.time() > deadline - 25.0:
                break
            for ps in [1.5, 2.0, 2.5, 3.0]:
                if time.time() > deadline - 25.0:
                    break
                dp = 1.0 / (1.0 + ps * norm_demands)
                H_cand = (1.0 / (dm_safe ** gamma)) * dp[np.newaxis, :]
                try_candidate(H_cand, f"fine2 g={gamma:.3f} ps={ps}")

    # Final fallback
    if best_H is None:
        best_H = (1.0 / (dm_safe ** 1.5)) * demand_penalty[np.newaxis, :]
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    # Ensure output shape matches input and all values are valid
    assert best_H.shape == dm.shape, "Shape mismatch"
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    return best_H

# EVOLVE-BLOCK-END
