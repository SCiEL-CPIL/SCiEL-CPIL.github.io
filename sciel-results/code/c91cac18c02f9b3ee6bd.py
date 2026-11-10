"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Returns a matrix of the same shape as distance_matrix where larger = more attractive.
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)

    # Base heuristic: inverse distance (larger = closer = more attractive)
    # Add a small epsilon to avoid division by zero
    base = 1.0 / (dm + 1e-9)

    # Penalize edges that would likely violate capacity when combined with demand
    # For each edge i->j, if demand[j] is large, make it less attractive
    # This is a soft penalty: multiply by a factor that decreases with demand
    demand_penalty = 1.0 / (1.0 + dem[None, :] * 0.1)
    heuristic = base * demand_penalty

    # Ensure all entries are positive
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9

    return heuristic


def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Run ACO simulation and return the best total route distance found.
    """
    n = len(demands)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    h = np.asarray(heuristic, dtype=np.float64)

    # Ensure heuristic is valid
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)

    # Pheromone matrix, start at 1
    tau = np.ones((n, n), dtype=np.float64)

    best_cost = np.inf

    rng = np.random.default_rng(seed)

    for it in range(n_iters):
        iter_best = np.inf

        for ant in range(n_ants):
            visited = [0]
            remaining_cap = cap
            total_dist = 0.0

            while True:
                current = visited[-1]
                # Find unvisited customers that fit in remaining capacity
                unvisited = [j for j in range(1, n) if j not in visited]
                feasible = [j for j in unvisited if dem[j] <= remaining_cap]

                if not feasible:
                    # Return to depot
                    total_dist += dm[current, 0]
                    visited.append(0)
                    remaining_cap = cap
                    if len(visited) >= n:
                        break
                    continue

                # Compute probabilities proportional to tau[i,j] * h[i,j]
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = tau[current, j] * h[current, j]

                total_prob = probs.sum()
                if total_prob <= 0 or not np.isfinite(total_prob):
                    # Fallback: pick the closest feasible node
                    dists = [dm[current, j] for j in feasible]
                    j = feasible[int(np.argmin(dists))]
                else:
                    probs = probs / total_prob
                    idx = rng.choice(len(feasible), p=probs)
                    j = feasible[idx]

                total_dist += dm[current, j]
                visited.append(j)
                remaining_cap -= dem[j]

                # Check if all customers are visited
                if len(visited) >= n:
                    break

            # Make sure we end at depot if we haven't already
            if visited[-1] != 0:
                total_dist += dm[visited[-1], 0]

            if total_dist < iter_best:
                iter_best = total_dist

        # Update pheromone
        tau *= 0.9
        if np.isfinite(iter_best) and iter_best > 0:
            tau += 1.0 / iter_best

        if iter_best < best_cost:
            best_cost = iter_best

    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Public entry point. Computes heuristic and runs ACO, but the task says
    the heuristic is returned. The ACO uses the heuristic internally.
    However, the scoring is based on the ACO result, not the heuristic matrix itself.
    
    Wait, re-reading the task: "Write heuristics(...) which returns an edge-attractiveness matrix".
    The scoring is "mean over instances of the lowest total route distance found by any ant."
    
    This means the evaluator calls heuristics() to get the matrix, then runs the ACO using that matrix.
    So I just need to return a good heuristic matrix. The ACO is run by the evaluator.
    
    Let me re-read: "The heuristic is computed once per instance, before the colony runs."
    
    So I just return the heuristic matrix. The evaluator handles the ACO.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    n = dm.shape[0]

    # Inverse distance as base
    base = 1.0 / (dm + 1e-9)

    # Demand-aware penalty: edges to high-demand nodes are slightly less attractive
    # This helps ants avoid getting stuck with a heavy load
    demand_factor = 1.0 / (1.0 + dem[None, :] * 0.05)
    heuristic = base * demand_factor

    # Ensure positivity and finiteness
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9

    return heuristic
# EVOLVE-BLOCK-END
