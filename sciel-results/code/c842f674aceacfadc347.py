import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for the ACO-CVRP instances.
    Uses a multi-stage search: coarse gamma grid, then fine grid,
    with a log-sum-exp based soft-min objective evaluated by a
    small ACO proxy, plus a final high-fidelity polish.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)

    def make_eta(gamma):
        """Compute heuristic matrix for a given exponent gamma."""
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def run_aco(eta, num_ants, num_iters, seed):
        """
        Run a small ACO and return the best total route distance found.
        Also returns a list of iteration bests for soft-min use.
        """
        rng = np.random.RandomState(seed)
        best_cost = float('inf')
        iter_bests = []

        # Pheromone initialization
        tau = np.ones_like(D)

        for it in range(num_iters):
            iter_best = float('inf')

            for ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_distance = 0.0
                unvisited = list(range(1, n))

                while unvisited:
                    rem_cap = capacity - current_load
                    feasible = [j for j in unvisited if demands[j] <= rem_cap]

                    if not feasible:
                        total_distance += D[current, 0]
                        break

                    if len(feasible) == 1:
                        next_node = feasible[0]
                    else:
                        f_arr = np.array(feasible)
                        weights = tau[current, f_arr] * eta[current, f_arr]
                        total_w = np.sum(weights)
                        if total_w <= 0:
                            next_node = f_arr[rng.randint(len(f_arr))]
                        else:
                            probs = weights / total_w
                            next_node = f_arr[rng.choice(len(f_arr), p=probs)]

                    total_distance += D[current, next_node]
                    current_load += demands[next_node]
                    unvisited.remove(next_node)
                    current = next_node

                    if not unvisited:
                        total_distance += D[current, 0]
                        break

                if total_distance < iter_best:
                    iter_best = total_distance

            if iter_best < best_cost:
                best_cost = iter_best
            iter_bests.append(iter_best)

            # Pheromone update
            tau *= 0.9
            if iter_best > 0:
                tau += 1.0 / iter_best

        return best_cost, iter_bests

    def soft_min_objective(gamma, num_ants, num_iters, seed, beta=10.0):
        """
        Evaluate gamma using soft-min over iteration bests to get a
        smooth surrogate for the min objective.
        """
        eta = make_eta(gamma)
        _, iter_bests = run_aco(eta, num_ants, num_iters, seed)
        if not iter_bests:
            return float('inf')
        costs = np.array(iter_bests, dtype=np.float64)
        # Soft-min: -beta * log(sum(exp(-costs/beta)))
        # This approximates min(costs) as beta -> inf
        m = np.min(costs)
        return float(m - beta * np.log(np.sum(np.exp(-(costs - m) / beta))))

    # Phase 1: Coarse search over gamma
    best_gamma = 1.0
    best_val = float('inf')
    coarse_gammas = np.arange(0.5, 3.5, 0.25)

    for gamma in coarse_gammas:
        if time.time() > deadline - 30.0:
            break
        val = soft_min_objective(gamma, 5, 5, seed=int(gamma * 1000))
        if val < best_val:
            best_val = val
            best_gamma = gamma

    # Phase 2: Fine search around best_gamma
    if time.time() < deadline - 20.0:
        fine_gammas = np.arange(best_gamma - 0.3, best_gamma + 0.3, 0.05)
        fine_gammas = fine_gammas[(fine_gammas > 0.1) & (fine_gammas < 5.0)]

        for gamma in fine_gammas:
            if time.time() > deadline - 15.0:
                break
            val = soft_min_objective(gamma, 8, 8, seed=int(gamma * 1000))
            if val < best_val:
                best_val = val
                best_gamma = gamma

    # Phase 3: Very fine search
    if time.time() < deadline - 10.0:
        vfg = np.arange(best_gamma - 0.08, best_gamma + 0.08, 0.01)
        vfg = vfg[(vfg > 0.1) & (vfg < 5.0)]

        for gamma in vfg:
            if time.time() > deadline - 8.0:
                break
            val = soft_min_objective(gamma, 10, 10, seed=int(gamma * 1000))
            if val < best_val:
                best_val = val
                best_gamma = gamma

    # Phase 4: Polish with higher fidelity ACO
    if time.time() < deadline - 5.0:
        polish_gammas = np.arange(best_gamma - 0.05, best_gamma + 0.05, 0.025)
        polish_gammas = polish_gammas[(polish_gammas > 0.1) & (polish_gammas < 5.0)]

        for gamma in polish_gammas:
            if time.time() > deadline - 3.0:
                break
            eta = make_eta(gamma)
            best_c, _ = run_aco(eta, 12, 12, seed=int(gamma * 1000))
            if best_c < best_val:
                best_val = best_c
                best_gamma = gamma

    # Final heuristic matrix
    final_eta = make_eta(best_gamma)

    # Safety checks
    if final_eta.shape != D.shape:
        final_eta = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_eta, 1e-9)

    final_eta = np.where(np.isfinite(final_eta) & (final_eta > 0), final_eta, 1e-9)

    return final_eta
