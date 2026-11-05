"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy independent set selection.
    Returns a score per vertex; higher score = selected earlier.
    Uses spectral centrality (principal eigenvector) + degree + neighborhood expansion.
    Vertices with lower combined centrality score are preferred (assigned higher priority).
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit

    # Convert to float for computations
    A = adjacency.astype(float)

    # Degree
    degree = A.sum(axis=1)

    # Power iteration for principal eigenvector (spectral centrality)
    # Use a moderate number of iterations to stay within time budget
    max_iter = 200
    v = np.ones(n) / np.sqrt(n)
    v = v / (np.linalg.norm(v) + 1e-12)

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        # Check convergence
        if np.linalg.norm(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new

    eigenvector = v  # principal eigenvector centrality

    # Neighborhood expansion: for each vertex, compute the sum of degrees of its neighbors
    # This captures how "expansive" the neighborhood is
    # nbr_degree_sum = A @ degree
    nbr_degree_sum = A @ degree

    # Outward expansion: number of unique neighbors' neighbors (approximated by sum of neighbor degrees / degree)
    # For vertices with degree > 0
    outward_exp = np.zeros(n)
    nonzero_deg = degree > 0
    outward_exp[nonzero_deg] = nbr_degree_sum[nonzero_deg] / degree[nonzero_deg]

    # Minimum neighbor degree: min degree among neighbors (or 0 if isolated)
    min_nbr_deg = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            neighbors = np.where(adjacency[i])[0]
            min_nbr_deg[i] = np.min(degree[neighbors])
        # else stays 0

    # Combine features into a single centrality score (higher = more central = less preferred)
    # We want to select vertices that are peripheral/less entangled
    # Normalize each feature to [0, 1] range
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_degree = normalize(degree)
    norm_eigenvector = normalize(eigenvector)
    norm_nbr_deg_sum = normalize(nbr_degree_sum)
    norm_outward = normalize(outward_exp)
    norm_min_nbr = normalize(min_nbr_deg)

    # Weighted combination: lower centrality = higher priority
    # Weights tuned to balance local and global structural information
    w1 = 0.35  # degree
    w2 = 0.35  # spectral centrality
    w3 = 0.10  # neighbor degree sum
    w4 = 0.10  # outward expansion
    w5 = 0.10  # min neighbor degree

    centrality = (
        w1 * norm_degree +
        w2 * norm_eigenvector +
        w3 * norm_nbr_deg_sum +
        w4 * norm_outward +
        w5 * norm_min_nbr
    )

    # Priority: higher score = selected first
    # We want to select vertices with LOW centrality first
    # So priority = -centrality, but we can add a small degree-based tiebreaker
    # Actually, let's just invert: priority = (1 - centrality)
    # This gives higher priority to peripheral nodes
    priority_scores = 1.0 - centrality

    return priority_scores
# EVOLVE-BLOCK-END
