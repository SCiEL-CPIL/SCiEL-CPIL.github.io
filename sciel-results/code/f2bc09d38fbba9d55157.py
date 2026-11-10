import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    deadline = time.time() + 295.0

    # Precompute
    demand_vec = demands.copy()
    demand_vec[0] = 0.0

    # Candidate parameter family:
    # params = [alpha, beta, gamma, delta]
    # heuristic[i,j] = (1/dist[i,j])^alpha * (1 + beta * cos_angle[i,j]) * (1 + gamma * depot_boost[i,j]) * (1 + delta * demand_scale[j])
    # where cos_angle is the angular continuity factor, depot_boost is 1 if i==0 or j==0 else 0,
    # demand_scale is normalized demand of j.

    # Precompute angular continuity matrix (cos of angle between depot->i and i->j)
    coord = coordinates
    cos_mat = np.ones((n, n))
    for i in range(1, n):
        v_i = coord[i] - coord[0]
        norm_v_i = np.linalg.norm(v_i)
        if norm_v_i < 1e-6:
            continue
        for j in range(1, n):
            if i == j:
                continue
            w_ij = coord[j] - coord[i]
            norm_w_ij = np.linalg.norm(w_ij)
            if norm_w_ij < 1e-6:
                continue
            dot = np.dot(v_i, w_ij)
            cos_val = np.clip(dot / (norm_v_i * norm_w_ij), -1.0, 1.0)
            cos_mat[i, j] = cos_val

    # Precompute depot boost matrix
    depot_boost = np.zeros((n, n))
    depot_boost[0, :] = 1.0
    depot_boost[:, 0] = 1.0

    # Precompute demand scale (normalized)
    max_demand = max(demand_vec.max(), 1e-9)
    demand_scale = demand_vec / max_demand

    def build_heuristic(params):
        alpha, beta, gamma, delta = params
        base = 1.0 / (distance_matrix + 1e-9) ** alpha
        angular = 1.0 + beta * cos_mat
        angular = np.maximum(angular, 0.1)
        dep = 1.0 + gamma * depot_boost
        dem = 1.0 + delta * demand_scale[None, :]
        h = base * angular * dep * dem
        return np.maximum(h, 1e-9)

    def run_aco(heuristic_mat, n_ants=20, n_iters=30, seed=42):
        """Run the exact ACO described in the task. Returns best total route distance."""
        rng = np.random.RandomState(seed)
        tau = np.ones((n, n))
        best_cost = float('inf')

        for _ in range(n_iters):
            iter_best = float('inf')
            for _ant in range(n_ants):
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                current = 0
                load = 0.0
                total_dist = 0.0
                remaining = n - 1

                while remaining > 0:
                    feasible = ~visited
                    feasible[0] = False
                    feasible = feasible & (load + demand_vec <= capacity)
                    feasible_idx = np.where(feasible)[0]

                    if len(feasible_idx) == 0:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        load = 0.0
                        continue

                    probs = tau[current, feasible_idx] * heuristic_mat[current, feasible_idx]
                    probs = np.maximum(probs, 1e-9)
                    probs /= probs.sum()
                    next_j = feasible_idx[rng.choice(len(feasible_idx), p=probs)]

                    total_dist += distance_matrix[current, next_j]
                    visited[next_j] = True
                    load += demand_vec[next_j]
                    current = next_j
                    remaining -= 1

                total_dist += distance_matrix[current, 0]

                if total_dist < iter_best:
                    iter_best = total_dist

            # Pheromone update
            tau *= 0.9
            if iter_best > 0:
                tau += 1.0 / iter_best

            if iter_best < best_cost:
                best_cost = iter_best

        return best_cost

    def objective(params):
        if time.time() > deadline:
            return float('inf')
        try:
            h = build_heuristic(params)
            # Use a reduced ACO for speed during optimization
            cost = run_aco(h, n_ants=10, n_iters=10, seed=123)
            return cost
        except Exception:
            return float('inf')

    # Nelder-Mead optimization over the 4 parameters
    # Initial guess: alpha=1, beta=0.5, gamma=0.3, delta=0.1
    x0 = np.array([1.0, 0.5, 0.3, 0.1])
    bounds = [(0.3, 3.0), (-0.8, 1.5), (0.0, 2.0), (-0.5, 1.5)]

    best_params = x0.copy()
    best_obj = float('inf')

    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 80, 'xatol': 0.01, 'fatol': 0.01,
                                   'adaptive': True})
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass

    # Clip to bounds
    for i, (lo, hi) in enumerate(bounds):
        best_params[i] = np.clip(best_params[i], lo, hi)

    # Final verification with full ACO
    final_heuristic = build_heuristic(best_params)

    # Ensure positivity and finite
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    if not np.all(np.isfinite(final_heuristic)):
        final_heuristic = 1.0 / (distance_matrix + 1e-9)

    return final_heuristic
