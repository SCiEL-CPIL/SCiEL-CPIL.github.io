# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Multi-criteria heuristic for ACO CVRP combining:
      1. Inverse distance (exponent -1.5)
      2. Quadratic demand penalty relative to capacity
      3. Directional bias toward the centroid of high-demand customers
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)

    n = distance_matrix.shape[0]

    # --- Component 1: inverse distance with exponent 1.5 ---
    eps = 1e-9
    dist = np.maximum(distance_matrix, eps)
    inv_dist = 1.0 / (dist ** 1.5)

    # --- Component 2: quadratic demand penalty ---
    # Heavier customers are penalised more strongly; scale by capacity
    demand_norm = demands / capacity  # in [0, 0.18] roughly
    demand_penalty = 1.0 / (1.0 + demand_norm ** 2)

    # --- Component 3: directional bias ---
    # Centroid of customers whose demand is above the median (excluding depot)
    customer_demands = demands[1:]
    median_demand = np.median(customer_demands)
    high_demand_mask = customer_demands > median_demand
    high_demand_indices = np.where(high_demand_mask)[0] + 1

    if len(high_demand_indices) > 0:
        target_centroid = coordinates[high_demand_indices].mean(axis=0)
    else:
        target_centroid = coordinates[1:].mean(axis=0)

    # For each edge i->j, compute alignment of direction (i->j) with direction (i->target_centroid)
    # This gives a gentle preference for moving toward the cluster of heavy customers
    dirs = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # (n, n, 2)
    to_target = target_centroid[np.newaxis, np.newaxis, :] - coordinates[:, np.newaxis, :]  # (n, 1, 2)

    # Normalise
    dir_norms = np.linalg.norm(dirs, axis=2, keepdims=True) + eps
    target_norms = np.linalg.norm(to_target, axis=2, keepdims=True) + eps

    dirs_unit = dirs / dir_norms
    target_unit = to_target / target_norms

    # Cosine similarity: (n, n, 1) -> (n, n)
    cos_sim = np.sum(dirs_unit * target_unit, axis=2)  # in [-1, 1]

    # Map cosine to a mild multiplier in [0.7, 1.3]
    directional_factor = 1.0 + 0.3 * cos_sim

    # --- Combine all components ---
    eta = inv_dist * demand_penalty * directional_factor

    # Ensure positivity and finiteness
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)

    # Diagonal: self-loops are useless
    np.fill_diagonal(eta, 1e-9)

    return eta
# EVOLVE-BLOCK-END
