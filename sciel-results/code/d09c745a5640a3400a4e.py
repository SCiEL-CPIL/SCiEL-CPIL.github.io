"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = adjacency.shape[0]
    adjacency_bool = adjacency.astype(bool)

    # Precompute neighbor lists and degrees
    degrees = adjacency_bool.sum(axis=1).astype(np.int64)
    neighbor_lists = [np.where(adjacency_bool[i])[0] for i in range(n)]
    neighbor_sets = [set(neighbor_lists[i]) for i in range(n)]

    def simulate_greedy(order):
        """Simulate greedy decoder given an explicit vertex order (descending priority)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                nbrs = neighbor_lists[v]
                if len(nbrs) > 0:
                    selected[nbrs] = True
        return count

    def simulate_greedy_fast(order):
        """Faster simulation using a blocked set approach."""
        blocked = set()
        count = 0
        for v in order:
            if v not in blocked:
                count += 1
                blocked.add(v)
                blocked.update(neighbor_lists[v])
        return count

    rng = np.random.RandomState(42)

    # Generate several initial orderings
    initial_orders = []

    # 1. Ascending degree (low degree first) - classic greedy MIS heuristic
    initial_orders.append(np.argsort(degrees))

    # 2. Descending degree
    initial_orders.append(np.argsort(-degrees))

    # 3. Random
    initial_orders.append(rng.permutation(n))

    # 4. Degree with small random noise
    deg_noise = degrees.astype(float) + rng.uniform(0, 1, n) * 0.01
    initial_orders.append(np.argsort(deg_noise))

    # 5. Another random
    initial_orders.append(rng.permutation(n))

    # 6. Core-based ordering: sort by (degree, second-degree)
    # Compute second-degree (sum of degrees of neighbors)
    second_deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if len(neighbor_lists[i]) > 0:
            second_deg[i] = degrees[neighbor_lists[i]].sum()

    # Sort by second_degree ascending, then degree ascending
    combined = second_deg * 1000 + degrees
    initial_orders.append(np.argsort(combined))

    # 7. Power iteration based score (like PageRank on the graph)
    # Use a few iterations of power method to get a centrality score
    try:
        if n > 0 and adjacency_bool.any():
            # Normalize by out-degree (which equals in-degree for undirected)
            deg_float = degrees.astype(float)
            deg_float[deg_float == 0] = 1.0  # avoid division by zero
            P = adjacency_bool.astype(float) / deg_float[:, np.newaxis]
            # Add a small teleportation to handle disconnected components
            alpha = 0.85
            P = alpha * P + (1 - alpha) / n
            scores_pr = np.ones(n) / n
            for _ in range(10):
                scores_pr = P.T @ scores_pr
                scores_pr /= scores_pr.sum()
            initial_orders.append(np.argsort(-scores_pr))
    except:
        pass

    # 8. Smooth surrogate: use soft-min of degrees of neighbors
    # Vertices with neighbors that have low degree are "easier" to include
    try:
        soft_min_scores = np.zeros(n)
        for i in range(n):
            nbrs = neighbor_lists[i]
            if len(nbrs) == 0:
                soft_min_scores[i] = -1.0  # isolated, easy to pick
            else:
                nbr_degrees = degrees[nbrs].astype(float)
                # Soft minimum via log-sum-exp
                min_val = nbr_degrees.min()
                log_sum_exp = min_val + np.log(np.sum(np.exp(nbr_degrees - min_val)))
                soft_min_scores[i] = log_sum_exp
        initial_orders.append(np.argsort(soft_min_scores))
    except:
        pass

    best_order = initial_orders[0].copy()
    best_size = simulate_greedy(best_order)

    # Local search on each initial ordering
    for init_idx, init_order in enumerate(initial_orders):
        if time.time() > deadline - 10:
            break

        curr_order = init_order.copy()
        curr_size = simulate_greedy(curr_order)

        # Phase 1: Adjacent swap local search (multiple passes)
        improved = True
        max_passes = 5
        for _pass in range(max_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                # Swap i and i+1
                new_order = curr_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                new_size = simulate_greedy(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True
            if not improved:
                break

        # Phase 2: Random swap local search to escape local optima
        no_improve = 0
        max_no_improve = 200
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            new_order = curr_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_size = simulate_greedy(new_order)
            if new_size > curr_size:
                curr_size = new_size
                curr_order = new_order
                no_improve = 0
            else:
                no_improve += 1

        if curr_size > best_size:
            best_size = curr_size
            best_order = curr_order

    # Phase 3: Final polish with more aggressive random restarts if time permits
    if time.time() < deadline - 15:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Start from best_order with random perturbations
            curr_order = best_order.copy()
            # Apply several random swaps
            for _ in range(10):
                i, j = rng.randint(0, n, 2)
                curr_order[i], curr_order[j] = curr_order[j], curr_order[i]
            curr_size = simulate_greedy(curr_order)

            # Local search
            improved = True
            for _pass in range(3):
                if time.time() > deadline - 10:
                    break
                improved = False
                for i in range(n - 1):
                    if time.time() > deadline - 10:
                        break
                    new_order = curr_order.copy()
                    new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                    new_size = simulate_greedy(new_order)
                    if new_size > curr_size:
                        curr_size = new_size
                        curr_order = new_order
                        improved = True
                if not improved:
                    break

            if curr_size > best_size:
                best_size = curr_size
                best_order = curr_order

    # Convert order to scores: earlier in order = higher score
    scores = np.zeros(n)
    for pos, v in enumerate(best_order):
        scores[v] = -pos

    return scores
# EVOLVE-BLOCK-END
