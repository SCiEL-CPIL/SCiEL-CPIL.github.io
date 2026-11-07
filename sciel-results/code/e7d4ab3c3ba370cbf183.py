# EVOLVE-BLOCK-START
import numpy as np
import time


def _fast_aco_sim(dm, demands, capacity, H, n_ants=20, n_iterations=30, seed=42, time_limit=2.0):
    """
    Fast simulation of the evaluator's ACO to estimate heuristic quality.
    Mirrors the evaluator: 20 ants, 30 iters, tau=0.9*tau+1/best.
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


def _make_heuristic(dm_safe, demands, gamma, demand_weight):
    """
    Build a multi-criteria heuristic:
      eta[i,j] = d(i,j)^(-gamma) * (1 / (1 + demand_weight * demand[j]))
    """
    n = dm_safe.shape[0]
    inv_dist = 1.0 / (dm_safe ** gamma)
    
    # Demand penalty: higher demand nodes get lower attractiveness
    # demand[j] is the demand of the destination node j
    demand_penalty = 1.0 / (1.0 + demand_weight * demands)
    demand_penalty[0] = 1.0  # Depot has no demand penalty
    
    # Broadcast: inv_dist[i,j] * demand_penalty[j]
    eta = inv_dist * demand_penalty[np.newaxis, :]
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    np.fill_diagonal(eta, 1e-9)
    
    return eta


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO of capacitated vehicle routing.
    Uses a demand-aware, multi-criteria heuristic combining inverse distance
    with a penalty for high-demand nodes. Searches over gamma and demand_weight.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill

    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]

    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    best_gamma = 1.0
    best_dw = 0.0
    best_avg_cost = np.inf
    best_H = None

    # Phase 1: Coarse search over gamma and demand_weight
    gammas_coarse = np.linspace(0.5, 4.0, 15)
    dws_coarse = np.array([0.0, 0.1, 0.2, 0.5, 1.0])

    for g in gammas_coarse:
        for dw in dws_coarse:
            if time.time() > deadline - 60.0:
                break

            H_candidate = _make_heuristic(dm_safe, demands, g, dw)

            costs = []
            for s in [0, 1, 2]:
                cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                     n_ants=10, n_iterations=10, seed=s, time_limit=1.5)
                if cost < np.inf:
                    costs.append(cost)

            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_avg_cost:
                    best_avg_cost = avg_cost
                    best_gamma = g
                    best_dw = dw
                    best_H = H_candidate

    # Phase 2: Fine-tune around the best (gamma, dw)
    if time.time() < deadline - 50.0:
        gammas_fine = np.linspace(max(0.1, best_gamma - 0.3), best_gamma + 0.3, 7)
        dws_fine = np.linspace(max(0.0, best_dw - 0.2), best_dw + 0.2, 5)

        for g in gammas_fine:
            for dw in dws_fine:
                if time.time() > deadline - 30.0:
                    break

                H_candidate = _make_heuristic(dm_safe, demands, g, dw)

                costs = []
                for s in [0, 1, 2, 3]:
                    cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                         n_ants=15, n_iterations=15, seed=s, time_limit=1.5)
                    if cost < np.inf:
                        costs.append(cost)

                if costs:
                    avg_cost = np.mean(costs)
                    if avg_cost < best_avg_cost:
                        best_avg_cost = avg_cost
                        best_gamma = g
                        best_dw = dw
                        best_H = H_candidate

    # Phase 3: Ultra-fine polish
    if time.time() < deadline - 25.0:
        gammas_ultra = np.linspace(max(0.1, best_gamma - 0.1), best_gamma + 0.1, 5)
        dws_ultra = np.linspace(max(0.0, best_dw - 0.1), best_dw + 0.1, 3)

        for g in gammas_ultra:
            for dw in dws_ultra:
                if time.time() > deadline - 15.0:
                    break

                H_candidate = _make_heuristic(dm_safe, demands, g, dw)

                costs = []
                for s in [0, 1, 2, 3, 4]:
                    cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                         n_ants=20, n_iterations=20, seed=s, time_limit=2.0)
                    if cost < np.inf:
                        costs.append(cost)

                if costs:
                    avg_cost = np.mean(costs)
                    if avg_cost < best_avg_cost:
                        best_avg_cost = avg_cost
                        best_gamma = g
                        best_dw = dw
                        best_H = H_candidate

    # Fallback
    if best_H is None:
        best_H = _make_heuristic(dm_safe, demands, 1.0, 0.0)

    return best_H

# EVOLVE-BLOCK-END
