import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    BIG = 1e9

    # --- Spectral approach: Fiedler vector ---
    D = np.diag(deg)
    L = D - A_float

    try:
        eigvals, eigvecs = np.linalg.eigh(L)

        positive_mask = eigvals > 1e-10
        if positive_mask.any():
            start_idx = np.where(positive_mask)[0][0]
            fiedler_vec = eigvecs[:, start_idx]
        else:
            fiedler_vec = np.zeros(n)

        fiedler_mag = np.abs(fiedler_vec)
        fiedler_max = fiedler_mag.max()
        if fiedler_max > 1e-12:
            fiedler_norm = fiedler_mag / fiedler_max
        else:
            fiedler_norm = np.zeros(n)

        num_modes = min(5, n - start_idx) if positive_mask.any() else 0
        if num_modes > 0:
            modes = eigvecs[:, start_idx:start_idx + num_modes]
            mode_weights = np.exp(-np.arange(num_modes) * 0.5)
            mode_weights /= mode_weights.sum()
            spectral_score = np.sum((modes ** 2) * mode_weights, axis=1)
            s_max = spectral_score.max()
            if s_max > 1e-12:
                spectral_score = spectral_score / s_max
            else:
                spectral_score = np.zeros(n)
        else:
            spectral_score = np.zeros(n)

    except Exception:
        fiedler_norm = np.zeros(n)
        spectral_score = np.zeros(n)

    # --- Fixed-point safety score ---
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    for it in range(300):
        if time.time() > deadline - 10.0:
            break
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        if np.allclose(new_s, s, rtol=1e-8, atol=1e-10):
            s = new_s
            break
        s = new_s

    # --- Normalize safety score ---
    s_noniso = s[has_edge]
    if s_noniso.size > 0:
        s_min = s_noniso.min()
        s_max_s = s_max_s - s_min if (s_max_s := s_noniso.max()) > s_min else 0
        if s_max_s > 1e-12:
            s_norm = (s - s_min) / s_max_s
        else:
            s_norm = np.zeros_like(s)
    else:
        s_norm = np.zeros_like(s)

    # --- Bipartite matching-inspired hub-leaf analysis ---
    # Identify "hubs" (high degree) and "leaves" (low degree neighbors)
    # A hub that covers many leaves is "essential" for blocking
    # We want to prioritize hubs that cover many unique leaves

    if n > 0:
        # Classify vertices: hubs have degree above median, leaves below
        deg_median = np.median(deg[has_edge]) if has_edge.any() else 0
        hub_mask = has_edge & (deg > deg_median)
        leaf_mask = has_edge & (deg <= deg_median)

        # For each hub, count how many leaf neighbors it has
        hub_leaf_count = np.zeros(n)
        if hub_mask.any() and leaf_mask.any():
            # hub_leaf_count[i] = number of leaf neighbors of hub i
            hub_leaf_count = (A_float @ leaf_mask.astype(float)).astype(int)

        # For each leaf, count how many hub neighbors it has
        leaf_hub_count = np.zeros(n)
        if hub_mask.any() and leaf_mask.any():
            leaf_hub_count = (A_float @ hub_mask.astype(float)).astype(int)

        # A hub is "essential" if it's the only hub neighbor for some leaves
        # Compute: for each hub, how many leaves have it as their ONLY hub neighbor
        essential_coverage = np.zeros(n)
        if hub_mask.any() and leaf_mask.any():
            # For each leaf, if it has exactly 1 hub neighbor, that hub gets credit
            only_hub_leaves = leaf_mask & (leaf_hub_count == 1)
            if only_hub_leaves.any():
                # For each such leaf, find its unique hub neighbor
                for leaf_idx in np.where(only_hub_leaves)[0]:
                    hub_neighbors = np.where(A[leaf_idx] & hub_mask)[0]
                    if len(hub_neighbors) > 0:
                        essential_coverage[hub_neighbors[0]] += 1

        # Normalize essential coverage
        if essential_coverage.max() > 0:
            essential_norm = essential_coverage / essential_coverage.max()
        else:
            essential_norm = np.zeros(n)

        # Also: hubs that cover many leaves total (not just essential)
        if hub_leaf_count.max() > 0:
            hub_leaf_norm = hub_leaf_count.astype(float) / hub_leaf_count.max()
        else:
            hub_leaf_norm = np.zeros(n)

    else:
        essential_norm = np.zeros(n)
        hub_leaf_norm = np.zeros(n)

    # --- Combine all signals ---
    alpha = 0.45   # Fiedler/peripheralness
    beta = 0.25    # safety score
    gamma = 0.10   # multi-mode spectral
    delta_w = 0.12 # essential hub coverage
    epsilon = 0.08 # hub-leaf total coverage

    scores = (alpha * fiedler_norm +
              beta * s_norm +
              gamma * spectral_score +
              delta_w * essential_norm +
              epsilon * hub_leaf_norm)

    # Isolated vertices must remain at top
    scores[~has_edge] = BIG

    # Small tie-breaker: lower degree slightly preferred
    if deg.max() > 0:
        deg_rank = -deg / (deg.max() + 1)
        scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
