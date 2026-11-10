import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    
    Uses a node-potential field derived from spatial density of demand:
    H[i,j] = dist^(-p) * exp(-beta * |coord[j] - centroid|) * (1/(1+a*dem[j]/cap))
    
    The centroid is the demand-weighted center of all customers.
    Grid search over (p, beta, a) with fast ACO simulator.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]

        if not np.all(np.isfinite(d)):
            return np.ones_like(d)

        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        eps = 1e-9
        coords = np.asarray(coordinates, dtype=np.float64)

        # Demand-weighted centroid of all customers (excluding depot)
        customer_dem = dem[1:]
        customer_coords = coords[1:]
        total_dem = customer_dem.sum()
        if total_dem > 0:
            centroid = (customer_coords * customer_dem[:, np.newaxis]).sum(axis=0) / total_dem
        else:
            centroid = customer_coords.mean(axis=0)

        # Distance from each node to the centroid
        dist_to_centroid = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
        max_dtc = dist_to_centroid.max() + eps

        def build_heuristic(p, beta, a):
            """
            H[i,j] = dist^(-p) * exp(-beta * dist_to_centroid[j]) * (1/(1+a*dem[j]/cap))
            """
            H = (1.0 / (d + eps) ** p).astype(np.float64)

            # Load-balancing pressure: decay exponentially with distance to centroid
            # Nodes closer to centroid are more attractive (higher potential)
            pressure = np.exp(-beta * dist_to_centroid)
            H = H * pressure[np.newaxis, :]

            # Demand inverse: lower demand -> higher attractiveness
            dem_factor = 1.0 / (1.0 + a * dem / cap)
            H = H * dem_factor[np.newaxis, :]

            H = np.maximum(H, eps)
            return H

        def simulate_aco(heuristic, seed=42, n_ants=5, n_iters=10):
            """
            Approximate the ACO: pheromone decay 0.9, reinforce with 1/best_cost.
            Returns best total distance found by any ant in any iteration.
            """
            rng = np.random.RandomState(seed)
            best_overall = float('inf')
            pher = np.ones((N, N), dtype=np.float64)

            for iteration in range(n_iters):
                iter_best = float('inf')

                for _ant in range(n_ants):
                    visited = np.zeros(N, dtype=bool)
                    visited[0] = True
                    current = 0
                    load = 0.0
                    total_dist = 0.0

                    while True:
                        # Find feasible customers
                        feasible = []
                        for j in range(1, N):
                            if not visited[j] and load + dem[j] <= cap:
                                feasible.append(j)

                        if not feasible:
                            # Return to depot
                            total_dist += d[current, 0]
                            current = 0
                            load = 0.0
                            if visited[1:].all():
                                break
                            continue

                        fe = np.array(feasible, dtype=int)
                        weights = pher[current, fe] * heuristic[current, fe]
                        weights = np.maximum(weights, 1e-9)
                        wsum = weights.sum()
                        if wsum <= 0:
                            probs = np.ones(len(fe)) / len(fe)
                        else:
                            probs = weights / wsum
                        next_j = fe[rng.choice(len(fe), p=probs)]

                        total_dist += d[current, next_j]
                        visited[next_j] = True
                        load += dem[next_j]
                        current = next_j

                    iter_best = min(iter_best, total_dist)
                    best_overall = min(best_overall, total_dist)

                # Update pheromone
                pher *= 0.9
                if iter_best > 0:
                    pher += 1.0 / iter_best

            return best_overall

        # Grid search over (p, beta, a)
        # p: distance power, beta: centroid decay, a: demand penalty
        p_vals = [0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
        beta_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        a_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

        best_cost = float('inf')
        best_params = (1.0, 0.0, 0.0)
        best_heuristic = build_heuristic(1.0, 0.0, 0.0).copy()

        eval_count = 0
        max_evals = 120  # Keep manageable within time budget

        # Coarse grid
        for p in p_vals:
            if time.time() > deadline - 30.0:
                break
            for beta in beta_vals:
                if time.time() > deadline - 30.0:
                    break
                for a in a_vals:
                    if time.time() > deadline - 30.0:
                        break
                    if eval_count >= max_evals:
                        break
                    try:
                        H = build_heuristic(p, beta, a)
                        c_val = simulate_aco(H, seed=42, n_ants=5, n_iters=10)
                        eval_count += 1
                        if np.isfinite(c_val) and c_val < best_cost:
                            best_cost = c_val
                            best_params = (p, beta, a)
                            best_heuristic = H.copy()
                    except Exception:
                        continue
                if time.time() > deadline - 30.0:
                    break
            if time.time() > deadline - 30.0:
                break

        # Local refinement around best_params
        if time.time() < deadline - 40.0:
            p0, beta0, a0 = best_params
            search_space = [
                (p0 * 0.9, beta0, a0), (p0 * 1.1, beta0, a0),
                (p0, max(0, beta0 - 0.5), a0), (p0, beta0 + 0.5, a0),
                (p0, beta0, max(0, a0 - 0.3)), (p0, beta0, a0 + 0.3),
                (p0 * 0.95, beta0 + 0.3, a0 + 0.2),
                (p0 * 1.05, max(0, beta0 - 0.3), a0 + 0.3),
                (p0, beta0 + 1.0, max(0, a0 - 0.5)),
                (p0 * 1.1, max(0, beta0 - 0.5), a0 + 0.5),
            ]
            for (p, beta, a) in search_space:
                if time.time() > deadline - 40.0:
                    break
                if p < 0.3 or p > 4.0 or beta < -0.1 or beta > 10.0 or a < -0.1 or a > 5.0:
                    continue
                try:
                    H = build_heuristic(p, beta, a)
                    c_val = simulate_aco(H, seed=42, n_ants=5, n_iters=10)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, beta, a)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # If time remains, do a second refinement with slightly different seeds
        if time.time() < deadline - 20.0:
            p0, beta0, a0 = best_params
            search_space2 = [
                (p0 * 0.85, beta0, a0), (p0 * 1.15, beta0, a0),
                (p0, max(0, beta0 - 1.0), a0), (p0, beta0 + 1.0, a0),
                (p0, beta0, max(0, a0 - 0.5)), (p0, beta0, a0 + 0.5),
                (p0 * 0.9, beta0 + 0.5, a0 - 0.2),
                (p0 * 1.1, max(0, beta0 - 0.5), a0 + 0.2),
            ]
            for (p, beta, a) in search_space2:
                if time.time() > deadline - 20.0:
                    break
                if p < 0.3 or p > 4.0 or beta < -0.1 or beta > 10.0 or a < -0.1 or a > 5.0:
                    continue
                try:
                    H = build_heuristic(p, beta, a)
                    # Use different seed for diversity
                    c_val = simulate_aco(H, seed=123, n_ants=5, n_iters=10)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, beta, a)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # Final safety: ensure all positive and finite
        if not np.all(np.isfinite(best_heuristic)):
            best_heuristic = build_heuristic(1.0, 0.0, 0.0).copy()
        best_heuristic = np.maximum(best_heuristic, eps)

        return best_heuristic

    except Exception:
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            H = 1.0 / (d + 1e-9)
            H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
            return H
        except:
            return np.ones((50, 50))
