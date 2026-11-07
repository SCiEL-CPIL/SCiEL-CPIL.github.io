"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _shortest_path_potential(distance_matrix):
    """Dense Dijkstra from depot (node 0) -> node potentials pi[i]."""
    n = distance_matrix.shape[0]
    pi = np.full(n, np.inf)
    pi[0] = 0.0
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        unvisited = ~visited
        if not unvisited.any():
            break
        unvisited_indices = np.where(unvisited)[0]
        min_idx = unvisited_indices[np.argmin(pi[unvisited_indices])]
        if pi[min_idx] == np.inf:
            break
        visited[min_idx] = True
        row = pi[min_idx] + distance_matrix[min_idx]
        new = np.where(unvisited, row, pi)
        pi = np.minimum(pi, new)
    pi = np.where(np.isfinite(pi), pi, distance_matrix.max())
    return pi


def _reduced_cost_matrix(distance_matrix):
    """Reduced cost c_ij = d_ij + pi_i - pi_j, floored at 1e-9."""
    n = distance_matrix.shape[0]
    pi = _shortest_path_potential(distance_matrix)
    rc = distance_matrix + pi[:, np.newaxis] - pi[np.newaxis, :]
    rc = np.maximum(rc, 1e-9)
    return rc


def _smooth_spatial_factor(coords, bandwidth):
    """Gaussian spatial proximity: exp(-d^2 / (2*sigma^2))."""
    n = coords.shape[0]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-dist_sq / (2.0 * bandwidth ** 2))


def _build_heuristic(dm, coords, demands, cap, gamma, bandwidth, demand_weight):
    """Build heuristic: reduced-cost base * Gaussian spatial * demand adjustment."""
    n = dm.shape[0]
    rc = _reduced_cost_matrix(dm)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        base_eta = 1.0 / np.power(rc + 1e-9, gamma)
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9
    
    spatial = _smooth_spatial_factor(coords, bandwidth)
    eta = base_eta * spatial
    
    if demand_weight > 0:
        dem = np.asarray(demands, dtype=np.float64)
        dem_norm = dem[1:] / 9.0
        dem_factor = np.clip(1.0 - demand_weight * dem_norm, 0.1, 1.0)
        eta[0, 1:] *= dem_factor
        eta[1:, 0] *= dem_factor
    
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    np.fill_diagonal(eta, 1e-9)
    return eta


def _proxy_aco(distance_matrix, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42, deadline=None):
    """Run a proxy ACO to evaluate a heuristic's quality (lower is better)."""
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]
    dm = distance_matrix
    eta = heuristic
    tau = np.ones((n, n))
    best_cost = np.inf
    demands_arr = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        iter_best = np.inf
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = np.zeros(n, dtype=bool)
            route_cost = 0.0
            while True:
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands_arr[j] <= cap - load:
                        feasible.append(j)
                if not feasible:
                    route_cost += dm[current, 0]
                    break
                farr = np.array(feasible, dtype=int)
                probs = tau[current, farr] * eta[current, farr]
                probs = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    probs = np.ones(len(farr))
                    total = probs.sum()
                probs /= total
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = int(np.searchsorted(cumsum, r))
                if idx >= len(farr):
                    idx = len(farr) - 1
                next_node = int(farr[idx])
                route_cost += dm[current, next_node]
                load += demands_arr[next_node]
                visited[next_node] = True
                current = next_node
            if route_cost < iter_best:
                iter_best = route_cost
        if iter_best < best_cost:
            best_cost = iter_best
        tau *= 0.9
        tau += 1.0 / max(iter_best, 1e-9)
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO of CVRP.
    
    Combines reduced-cost progress signal with smooth Gaussian spatial locality
    and demand-aware depot-edge weighting. Two-phase proxy ACO search.
    """
    start_time = time.time()
    deadline = start_time + 270.0
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Fallback
    rc = _reduced_cost_matrix(dm)
    with np.errstate(divide='ignore', invalid='ignore'):
        fallback = 1.0 / np.power(rc + 1e-9, 2.0)
    fallback[~np.isfinite(fallback)] = 1e-9
    fallback[fallback <= 0] = 1e-9
    np.fill_diagonal(fallback, 1e-9)
    
    seed = 42
    best_score = np.inf
    best_heuristic = fallback
    best_params = (2.0, 0.2, 0.0)
    
    # Phase 1: Coarse grid with fast proxy
    gamma_vals = [1.5, 2.0, 2.5]
    bw_vals = [0.12, 0.18, 0.25, 0.35]
    dw_vals = [0.0, 0.2, 0.4]
    
    for gamma in gamma_vals:
        for bw in bw_vals:
            for dw in dw_vals:
                if time.time() > deadline - 80.0:
                    break
                try:
                    eta = _build_heuristic(dm, coords, dem, cap, gamma, bw, dw)
                except Exception:
                    continue
                if eta.shape != dm.shape or not np.all(np.isfinite(eta)) or np.any(eta <= 0):
                    continue
                score = _proxy_aco(dm, dem, cap, eta, n_ants=10, n_iters=15,
                                   seed=seed, deadline=deadline - 40.0)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_heuristic = eta.copy()
                    best_params = (gamma, bw, dw)
    
    # Phase 2: Fine-tune around best
    if time.time() < deadline - 50.0:
        g0, bw0, dw0 = best_params
        fine_g = [g0 - 0.3, g0, g0 + 0.3]
        fine_bw = [max(0.05, bw0 - 0.06), bw0, bw0 + 0.06]
        fine_dw = [max(0.0, dw0 - 0.15), dw0, min(0.6, dw0 + 0.15)]
        
        for gamma in fine_g:
            for bw in fine_bw:
                for dw in fine_dw:
                    if time.time() > deadline - 15.0:
                        break
                    if gamma < 0.5 or bw < 0.05 or dw < 0:
                        continue
                    try:
                        eta = _build_heuristic(dm, coords, dem, cap, gamma, bw, dw)
                    except Exception:
                        continue
                    if eta.shape != dm.shape or not np.all(np.isfinite(eta)) or np.any(eta <= 0):
                        continue
                    score = _proxy_aco(dm, dem, cap, eta, n_ants=20, n_iters=30,
                                       seed=seed, deadline=deadline - 5.0)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_heuristic = eta.copy()
    
    if best_heuristic.shape != dm.shape:
        best_heuristic = fallback
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    np.fill_diagonal(best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
