import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for the ACO-CVRP instances.
    Strategy: Demand-aware multi-criteria heuristic.
    eta[i,j] = d(i,j)^-gamma * (1 / (1 + alpha * demand[j]))
    We search for optimal gamma and alpha using a fast ACO proxy.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)

    def make_eta(gamma, alpha):
        """
        Compute heuristic matrix for given distance exponent and demand penalty.
        """
        # Distance part
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_part = np.where(D > 0, D ** (-gamma), 1e-9)
        
        # Demand part: penalize high demand nodes
        # demand_factor[j] = 1 / (1 + alpha * demand[j])
        # We want eta[i,j] to be high for low demand j (if alpha > 0)
        # Wait, the inspiration says "penalty for high-demand nodes"
        # So high demand -> low attractiveness.
        # 1 / (1 + alpha * d_j) is smaller for larger d_j. Correct.
        
        demand_part = 1.0 / (1.0 + alpha * demands)
        
        # eta[i,j] = dist_part[i,j] * demand_part[j]
        eta = dist_part * demand_part[np.newaxis, :]
        
        # Ensure diagonal is 0 or small (not used in selection usually, but good practice)
        np.fill_diagonal(eta, 1e-9)
        
        # Safety: replace non-finite or non-positive with 1e-9
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta

    def run_aco(eta, num_ants, num_iters, seed):
        """
        Run a small ACO and return the best total route distance found.
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

    def soft_min_objective(gamma, alpha, num_ants, num_iters, seed, beta=10.0):
        """
        Evaluate gamma and alpha using soft-min over iteration bests.
        """
        eta = make_eta(gamma, alpha)
        _, iter_bests = run_aco(eta, num_ants, num_iters, seed)
        if not iter_bests:
            return float('inf')
        costs = np.array(iter_bests, dtype=np.float64)
        m = np.min(costs)
        return float(m - beta * np.log(np.sum(np.exp(-(costs - m) / beta))))

    # Phase 1: Coarse search over gamma and alpha
    best_gamma = 1.0
    best_alpha = 0.5
    best_val = float('inf')
    
    coarse_gammas = [0.5, 1.0, 1.5, 2.0]
    coarse_alphas = [0.0, 0.25, 0.5, 1.0]

    for gamma in coarse_gammas:
        if time.time() > deadline - 60.0:
            break
        for alpha in coarse_alphas:
            if time.time() > deadline - 60.0:
                break
            seed = int(gamma * 1000 + alpha * 100)
            val = soft_min_objective(gamma, alpha, 5, 5, seed)
            if val < best_val:
                best_val = val
                best_gamma = gamma
                best_alpha = alpha

    # Phase 2: Fine search around best
    if time.time() < deadline - 30.0:
        fine_gammas = np.arange(best_gamma - 0.2, best_gamma + 0.2, 0.1)
        fine_alphas = np.arange(max(0.0, best_alpha - 0.25), best_alpha + 0.25, 0.1)
        
        for gamma in fine_gammas:
            if time.time() > deadline - 30.0:
                break
            for alpha in fine_alphas:
                if time.time() > deadline - 30.0:
                    break
                if alpha < 0: continue
                seed = int(gamma * 1000 + alpha * 100)
                val = soft_min_objective(gamma, alpha, 8, 8, seed)
                if val < best_val:
                    best_val = val
                    best_gamma = gamma
                    best_alpha = alpha

    # Phase 3: Very fine search
    if time.time() < deadline - 10.0:
        vfg = np.arange(best_gamma - 0.05, best_gamma + 0.05, 0.02)
        vfa = np.arange(max(0.0, best_alpha - 0.1), best_alpha + 0.1, 0.05)
        
        for gamma in vfg:
            if time.time() > deadline - 10.0:
                break
            for alpha in vfa:
                if time.time() > deadline - 10.0:
                    break
                if alpha < 0: continue
                seed = int(gamma * 1000 + alpha * 100)
                val = soft_min_objective(gamma, alpha, 10, 10, seed)
                if val < best_val:
                    best_val = val
                    best_gamma = gamma
                    best_alpha = alpha

    # Final heuristic matrix
    final_eta = make_eta(best_gamma, best_alpha)

    # Safety checks
    if final_eta.shape != D.shape:
        final_eta = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_eta, 1e-9)

    final_eta = np.where(np.isfinite(final_eta) & (final_eta > 0), final_eta, 1e-9)

    return final_eta
