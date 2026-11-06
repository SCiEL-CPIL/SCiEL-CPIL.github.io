"""Maximum Cut via multi-restart local search with simulated annealing and Goemans-Williamson rounding."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-restart local search with simulated annealing and GW rounding.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

    # Deadline: leave 15 seconds margin for safety
    start_time = time.time()
    deadline = start_time + 135.0

    def time_left():
        return deadline - time.time()

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def compute_same(labels):
        """Compute for each vertex the number of neighbors in the same group."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        same = np.zeros(n, dtype=np.int32)
        mask0 = labels == 0
        mask1 = labels == 1
        if np.any(mask0):
            same[mask0] = (adj @ in_group0)[mask0]
        if np.any(mask1):
            same[mask1] = (adj @ in_group1)[mask1]
        return same

    def local_search(labels, same, deadline):
        """
        Run local search using a priority queue of positive-gain flips.
        """
        labels = labels.copy()
        same = same.copy()
        cut = compute_cut(labels)

        # Build initial priority queue of all vertices with positive gain
        # gain[v] = 2*same[v] - degrees[v]
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Process the best available flip
            while heap:
                neg_gain, v = heapq.heappop(heap)
                gain = -neg_gain

                # Verify gain is still valid (may have changed due to other flips)
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Flip vertex v
                old_group = labels[v]
                new_group = 1 - old_group
                labels[v] = new_group

                # Update same[v]
                if new_group == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors of v
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_group:
                        same[u] -= 1
                    elif labels[u] == new_group:
                        same[u] += 1

                    # Check if u now has positive gain
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0:
                        heapq.heappush(heap, (-u_gain, u))

                cut += current_gain
                improved = True
                break

        return labels, cut, same

    def simulated_annealing(labels, same, deadline, rng):
        """
        Run simulated annealing to escape local optima.
        """
        labels = labels.copy()
        same = same.copy()
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut
        best_same = same.copy()

        T = 50.0
        cooling = 0.95
        min_T = 0.1
        max_steps = 10000

        for step in range(max_steps):
            if time_left() <= 0:
                break
            if T < min_T:
                break

            # Pick a random vertex
            v = rng.integers(0, n)
            gain = 2 * same[v] - degrees[v]

            # Decide whether to accept
            if gain > 0 or (T > 0 and rng.random() < np.exp(gain / T)):
                # Flip vertex v
                old_group = labels[v]
                new_group = 1 - old_group
                labels[v] = new_group

                # Update same[v]
                if new_group == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_group:
                        same[u] -= 1
                    elif labels[u] == new_group:
                        same[u] += 1

                cut += gain
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
                    best_same = same.copy()

            T *= cooling

        # Final local search polish
        best_labels, best_cut, best_same = local_search(best_labels, best_same, deadline)
        return best_labels, best_cut, best_same

    def goemans_williamson_init(rng):
        """Generate a random vector for each vertex and round to get a partition."""
        # Generate random unit vectors in R^n
        vecs = rng.standard_normal((n, n))
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms

        # Generate random hyperplane normal
        normal = rng.standard_normal(n)
        normal /= (np.linalg.norm(normal) + 1e-10)

        # Project each vertex onto the normal
        projections = vecs @ normal

        # Round: if projection > 0, label 1; else label 0
        labels = (projections > 0).astype(np.int8)
        return labels

    def random_init(rng):
        return rng.integers(0, 2, size=n, dtype=np.int8)

    def greedy_init():
        """Greedy sequential initialization."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            nbrs = neighbors[v]
            if len(nbrs) == 0:
                continue
            count_0 = np.sum(labels[nbrs] == 0)
            count_1 = np.sum(labels[nbrs] == 1)
            if count_1 > count_0:
                labels[v] = 1
            else:
                labels[v] = 0
        return labels

    def alternating_init():
        return np.arange(n) % 2

    # Generate initializations
    rng = np.random.default_rng(42)

    initializations = []
    initializations.append(alternating_init())
    initializations.append(greedy_init())
    initializations.append(goemans_williamson_init(rng))
    initializations.append(goemans_williamson_init(rng))
    initializations.append(goemans_williamson_init(rng))

    for _ in range(30):
        initializations.append(random_init(rng))

    # All zeros with one flip
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_zero[0] = 1
    initializations.append(labels_zero.copy())

    # All ones with one flip
    labels_one = np.ones(n, dtype=np.int8)
    labels_one[0] = 0
    initializations.append(labels_one.copy())

    best_labels = None
    best_cut = -1
    best_same = None

    # Phase 1: Local search on all initializations
    for init_labels in initializations:
        if time_left() <= 10:
            break
        same = compute_same(init_labels)
        labels, cut, same = local_search(init_labels, same, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            best_same = same.copy()

    # Phase 2: Simulated annealing from best and some random starts
    if best_labels is not None and time_left() > 20:
        sa_labels, sa_cut, sa_same = simulated_annealing(best_labels, best_same, deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
            best_same = sa_same.copy()

    # Phase 3: More SA restarts if time permits
    sa_count = 0
    while time_left() > 15 and sa_count < 10:
        init = random_init(rng)
        same = compute_same(init)
        sa_labels, sa_cut, sa_same = simulated_annealing(init, same, deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
            best_same = sa_same.copy()
        sa_count += 1

    # Final polish
    if best_labels is not None and time_left() > 2:
        best_labels, best_cut, best_same = local_search(best_labels, best_same, deadline)

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
