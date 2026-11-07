"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix for a capacitated VRP using
    a composite heuristic: inverse distance, demand-aware term, and
    proximity to nearest customers (clustering bonus).
    """
    n = distance_matrix.shape[0]
    # Work with float copies to avoid integer overflow or type issues
    D = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)

    # Base: inverse distance (closer = more attractive)
    # Add small epsilon to avoid division by zero
    base = 1.0 / (D + 1e-9)

    # Demand-aware heuristic: prefer edges to customers with moderate demand
    # (high demand customers should be grouped, so edges to/from them can be
    # slightly less attractive to avoid splitting, but also not zero)
    # Use a term that rewards going to a customer whose demand leaves good
    # remaining capacity utilization. A simple proxy: 1/(demand[j] + 1)
    # multiplied into the base, but this could bias too much. Instead, add
    # a small penalty for very high-demand customers on the way in.
    # Actually, for ACO the heuristic is static. Let's use a blend:
    #   eta[i,j] = (1/d_ij)^alpha * (1/(1 + dem_j/capacity))^beta
    # with alpha=1, beta=0.5
    alpha = 1.0
    beta = 0.5
    dem_norm = dem / capacity  # normalized demand, 0 for depot
    # For the depot row, demand is 0, so the term is 1.0
    demand_factor = 1.0 / (1.0 + dem_norm) ** beta  # shape (n,)
    # Multiply each column j by demand_factor[j]
    # base is (n,n); we want eta[i,j] = base[i,j] * demand_factor[j]
    eta = base * demand_factor[np.newaxis, :]

    # Clustering bonus: prefer edges that keep the ant near already-visited
    # customers. Since the heuristic is static, we can't know visited state.
    # Instead, use a global measure: for each customer j, compute how "central"
    # it is. Customers that are close to many others get a slight bonus.
    # This encourages grouping nearby customers in the same route.
    # For each j, compute sum of 1/d(i,j) for i != j, normalized.
    # This is expensive for large n, but n=51 here.
    # Actually, let's use a simpler idea: compute for each pair (i,j) a bonus
    # if i and j are "similar" in terms of their neighborhood.
    # A cheaper approach: compute a density score for each node.
    density = np.zeros(n)
    for j in range(n):
        # Sum of inverse distances from all other nodes to j
        if n > 1:
            other = np.delete(D[:, j], j)
            density[j] = np.sum(1.0 / (other + 1e-9))
    # Normalize density to [0,1] roughly
    if density.max() > 0:
        density_norm = density / density.max()
    else:
        density_norm = np.ones(n)
    # Add a small clustering bonus: multiply by (1 + 0.3 * density_norm[j])
    cluster_bonus = 1.0 + 0.3 * density_norm[np.newaxis, :]
    eta = eta * cluster_bonus

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    return eta
# EVOLVE-BLOCK-END
