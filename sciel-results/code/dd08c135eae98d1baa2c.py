"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.cluster.vq import kmeans2


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
        cand = np.where(unvisited, row, np.inf)
        # only relax unvisited nodes
        new = np.where(unvisited, row, pi)
        pi = np.minimum(pi, new)
    pi = np.where(np.isfinite(pi), pi, distance_matrix.max())
    return pi


def _reduced_cost_matrix(distance_matrix):
    """Reduced cost c_ij = d_ij + pi_i - pi_j, floored at 1e-9.
    Small reduced cost means the edge is 'better than expected' given potentials."""
    n = distance_matrix.shape[0]
    pi = _shortest_path_potential(distance_matrix)
    rc = distance_matrix + pi[:, np.newaxis] - pi[np.newaxis, :]
    rc = np.maximum(rc, 1e-9)
    return rc


def _build_heuristic(distance_matrix, demands, capacity, n_clusters, gamma, cross_penalty):
    """Combine reduced-cost progress signal with cluster affinity."""
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    coords = None  # coordinates passed separately; handled in caller

    rc = _reduced_cost_matrix(dm)
    # distance-progressive base: reward edges with small reduced cost,
    # tempered by distance exponent gamma for long-range decay
    with np.errstate(divide='ignore', invalid='ignore'):
        base_eta = 1.0 / np.power(rc + 1e-9, gamma)
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9

    if n_clusters <= 1 or n_clusters >= n:
        np.fill_diagonal(base_eta, 1e-9)
        return base_eta
    return base_eta, n_clusters, cross_penalty


def _apply_cluster_penalty(base_eta, coordinates, n_clusters, cross_penalty, seed=42):
    """Penalize cross-cluster edges by cross_penalty (encourages spatial locality)."""
    n = base_eta.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    heuristic = base_eta.copy()

    if n_clusters <= 1 or n_clusters >= n:
        np.fill_diagonal(heuristic, 1e-9)
        return heuristic

    customer_coords = coords[1:]
    try:
        actual_k = min(n_clusters, n - 1)
        if actual_k <= 1:
            np.fill_diagonal(heuristic, 1e-9)
            return heuristic

        centroids, labels = kmeans2(customer_coords, actual_k, minit='points', seed=seed)

        # assign depot to nearest cluster center
        dists_to_centroids = np.linalg.norm(centroids - coords[0], axis=1)
        depot_cluster = int(np.argmin(dists_to_centroids))

        full_labels = np.zeros(n, dtype=int)
        full_labels[0] = depot_cluster
        full_labels[1:] = labels

        # vectorized cross-cluster penalty
        same = full_labels[:, np.newaxis] == full_labels[np.newaxis, :]
        mask = ~same
        np.fill_diagonal(mask, False)
        heuristic[mask] *= cross_penalty

        np.fill_diagonal(heuristic, 1e-9)
        heuristic[~np.isfinite(heuristic)] = 1e-9
        heuristic[heuristic <= 0] = 1e-9
        return heuristic
    except Exception:
        np.fill_diagonal(heuristic, 1e-9)
        return heuristic


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

    Structure: reduced-cost progress signal (reward edges that advance toward
    the depot under shortest-path potentials) combined with K-Means cluster
    affinity (encourage spatial locality / intra-cluster travel). A proxy ACO
    searches over (clusters, distance exponent, cross-cluster penalty).
    """
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 300s limit

    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)

    # fallback base heuristic (reduced-cost, gamma=2)
    rc = _reduced_cost_matrix(dm)
    with np.errstate(divide='ignore', invalid='ignore'):
        fallback = 1.0 / np.power(rc + 1e-9, 2.0)
    fallback[~np.isfinite(fallback)] = 1e-9
    fallback[fallback <= 0] = 1e-9
    np.fill_diagonal(fallback, 1e-9)

    total_demand = float(np.sum(dem[1:]))
    min_vehicles = max(1, int(np.ceil(total_demand / cap)))

    k_values = [min_vehicles, min_vehicles + 1, min_vehicles + 2]
    gamma_values = [1.5, 2.0, 2.5]
    cross_values = [0.05, 0.1, 0.2]

    seed = 42
    best_score = np.inf
    best_heuristic = fallback

    for k in k_values:
        for gamma in gamma_values:
            for cp in cross_values:
                if time.time() > deadline - 15.0:
                    break
                try:
                    base_eta, kk, ccp = _build_heuristic(dm, dem, cap, k, gamma, cp)
                    eta = _apply_cluster_penalty(base_eta, coords, kk, ccp, seed=seed)
                except Exception:
                    continue

                if eta.shape != dm.shape or not np.all(np.isfinite(eta)):
                    continue
                if np.any(eta <= 0):
                    continue

                score = _proxy_aco(dm, dem, cap, eta, n_ants=20, n_iters=30,
                                   seed=seed, deadline=deadline - 5.0)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_heuristic = eta.copy()

    # final validation
    if best_heuristic.shape != dm.shape:
        best_heuristic = fallback
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    np.fill_diagonal(best_heuristic, 1e-9)

    return best_heuristic
# EVOLVE-BLOCK-END
