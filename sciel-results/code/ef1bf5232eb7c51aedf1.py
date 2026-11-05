"""Rewritten program for Maximum Independent Set via priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbours of already picked vertices)
    yields a large independent set.

    Strategy:
      1. Convert adjacency to a numpy bool/int matrix for fast ops.
      2. Compute several structural features per vertex:
           - degree
           - number of neighbours that are high-degree (local clustering proxy)
           - eigenvalue-based centrality (principal eigenvector)
           - a 'loneliness' score: how few of its neighbours are also high-degree
      3. Combine into a single score where lower-score vertices are picked first
         (since the decoder picks in descending order, we return negative of a
         'cost' so that low-cost vertices have high priority).

    The key insight: vertices with low degree and few high-degree neighbours
    are 'safe' to pick early because they block few other vertices.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds

    A = np.asarray(adjacency, dtype=np.int8)
    n = A.shape[0]

    # Degree
    deg = A.sum(axis=1).astype(np.float64)

    # Check if graph is dense enough to need spectral methods
    # For sparse graphs, spectral methods are expensive; use simpler heuristics
    edge_count = deg.sum() / 2.0
    density = edge_count / (n * (n - 1) / 2.0) if n > 1 else 0.0

    # --- Feature 1: Degree (lower is better) ---
    # Normalize degree to [0, 1]
    deg_norm = deg / (deg.max() + 1e-9) if deg.max() > 0 else np.zeros(n)

    # --- Feature 2: Number of high-degree neighbours ---
    # A vertex with many high-degree neighbours is 'risky' to pick early
    # because its neighbours are also important.
    # Define "high-degree" as above median degree.
    med_deg = np.median(deg)
    high_deg_mask = deg > med_deg

    # Count how many neighbours of each vertex are high-degree
    # A @ high_deg_mask gives for each vertex i, the count of its neighbours j
    # where j is high-degree.
    high_deg_neigh_count = (A @ high_deg_mask.astype(np.int8)).astype(np.float64)
    # Normalize
    hdnc_norm = high_deg_neigh_count / (high_deg_neigh_count.max() + 1e-9) if high_deg_neigh_count.max() > 0 else np.zeros(n)

    # --- Feature 3: Spectral centrality (principal eigenvector) ---
    # Vertices with high spectral centrality are 'hubs' and should be picked
    # later (or not at all in favor of their many neighbours).
    # Use power iteration for the principal eigenvector of A.
    if n > 0 and density < 0.3:
        # Power iteration for top eigenvector
        v = np.random.rand(n)
        v /= np.linalg.norm(v) + 1e-9
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v = v_new / norm
        spectral = np.abs(v)
        spectral_norm = spectral / (spectral.max() + 1e-9) if spectral.max() > 0 else np.zeros(n)
    else:
        # For dense graphs, skip spectral (too expensive or less useful)
        spectral_norm = np.zeros(n)

    # --- Feature 4: 'Loneliness' - how isolated is this vertex from other hubs? ---
    # A vertex with low degree AND few high-degree neighbours is ideal to pick first.
    # Combine: low deg_norm + low hdnc_norm + low spectral_norm
    # We want to MINIMIZE this cost, so we return negative for the priority.
    cost = 0.5 * deg_norm + 0.3 * hdnc_norm + 0.2 * spectral_norm

    # Add a small random tie-breaker to avoid ties
    rng = np.random.default_rng(42)
    tie_break = rng.random(n) * 1e-6
    cost += tie_break

    # Return negative cost so that lower-cost vertices have higher priority
    return -cost
# EVOLVE-BLOCK-END
