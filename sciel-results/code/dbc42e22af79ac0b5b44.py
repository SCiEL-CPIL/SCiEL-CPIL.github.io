"""Maximum Cut via spectral initialization, batched local search, and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral initialization, batched local search, and simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.int32)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=int)

    # Precompute degrees and neighbor lists
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbor_lists = [np.where(adj[v] > 0)[0] for v in range(n)]

    start_time = time.time()
    deadline = start_time + 130.0  # Leave safety margin before 150s kill

    rng = np.random.default_rng(seed=42)

    def compute_cut(labels):
        """Compute total cut size."""
        in0 = (labels == 0).astype(np.int32)
        in1 = (labels == 1).astype(np.int32)
        return int(in0 @ adj @ in1)

    def compute_opp_count(labels):
        """For each vertex, count neighbors in the opposite partition."""
        opp = np.zeros(n, dtype=np.int32)
        for v in range(n):
            opp[v] = adj[v][labels != labels[v]].sum()
        return opp

    def spectral_init():
        """Generate initial labels using Fiedler vector of graph Laplacian."""
        try:
            L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            fiedler_centered = fiedler - np.mean(fiedler)
            labels = (fiedler_centered > 0).astype(np.int32)
            return labels
        except Exception:
            return rng.integers(0, 2, size=n).astype(np.int32)

    def greedy_init():
        """Greedy sequential initialization by degree order."""
        labels = np.zeros(n, dtype=np.int32)
        order = np.argsort(-degrees)
        for v in order:
            neigh = neighbor_lists[v]
            if len(neigh) == 0:
                continue
            count_0 = np.sum(labels[neigh] == 0)
            count_1 = np.sum(labels[neigh] == 1)
            if count_1 > count_0:
                labels[v] = 1
        return labels

    def batched_local_search(labels, deadline):
        """
        Run local search with batched non-conflicting flips.
        In each iteration, find all vertices with positive gain, apply all
        non-conflicting ones simultaneously, then recompute gains.
        """
        labels = labels.copy()
        opp_count = compute_opp_count(labels)
        current_cut = compute_cut(labels)

        while time.time() < deadline:
            # Compute gains for all vertices
            gains = degrees - 2 * opp_count
            # Find vertices with positive gain
            candidates = np.where(gains > 0)[0]
            if len(candidates) == 0:
                break

            # Find non-conflicting subset: no two candidates are adjacent
            # Greedy selection: sort by gain descending, pick if no neighbor already picked
            candidates_sorted = candidates[np.argsort(-gains[candidates])]
            to_flip = []
            flipped_set = set()
            for v in candidates_sorted:
                if v in flipped_set:
                    continue
                # Check if any neighbor is already in to_flip
                conflict = False
                for u in neighbor_lists[v]:
                    if u in flipped_set:
                        conflict = True
                        break
                if not conflict:
                    to_flip.append(v)
                    flipped_set.add(v)

            if not to_flip:
                break

            # Apply all flips in the batch
            for v in to_flip:
                labels[v] = 1 - labels[v]

            # Update opp_count for all affected vertices
            # Vertices affected: all to_flip vertices and all their neighbors
            affected = set(to_flip)
            for v in to_flip:
                for u in neighbor_lists[v]:
                    affected.add(u)

            # Recompute opp_count for affected vertices
            for v in affected:
                opp_count[v] = adj[v][labels != labels[v]].sum()

            # Recompute cut
            current_cut = compute_cut(labels)

            if time.time() > deadline:
                break

        return labels, current_cut

    def simulated_annealing(labels, deadline):
        """
        Run simulated annealing to escape local optima.
        """
        labels = labels.copy()
        opp_count = compute_opp_count(labels)
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut

        t0 = 30.0
        cooling = 0.9995
        temp = t0

        while time.time() < deadline and temp > 0.01:
            temp *= cooling

            # Pick a random vertex to flip
            v = rng.integers(0, n)
            old_label = labels[v]
            labels[v] = 1 - old_label

            # Compute delta in cut
            # When v flips: edges from v to same-group become cut, edges to opp-group stop being cut
            # delta = (degree[v] - 2*opp_count[v]) where opp_count is BEFORE flip
            delta = degrees[v] - 2 * opp_count[v]

            if delta > 0:
                cut += delta
                # Update opp_count for v and neighbors
                opp_count[v] = degrees[v] - opp_count[v]
                for u in neighbor_lists[v]:
                    if labels[u] == labels[v]:
                        opp_count[u] -= 1
                    else:
                        opp_count[u] += 1
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
            else:
                # Accept with probability exp(delta/temp)
                if rng.random() < np.exp(delta / max(temp, 1e-10)):
                    cut += delta
                    opp_count[v] = degrees[v] - opp_count[v]
                    for u in neighbor_lists[v]:
                        if labels[u] == labels[v]:
                            opp_count[u] -= 1
                        else:
                            opp_count[u] += 1
                else:
                    # Reject: revert
                    labels[v] = old_label

        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    # Generate diverse initializations
    initializations = []

    # Spectral initialization (highest priority)
    initializations.append(spectral_init())

    # Alternating
    initializations.append(np.arange(n) % 2)

    # Greedy
    initializations.append(greedy_init())

    # Random restarts
    for _ in range(8):
        initializations.append(rng.integers(0, 2, size=n).astype(np.int32))

    # All zeros with one flip
    lz = np.zeros(n, dtype=np.int32)
    lz[0] = 1
    initializations.append(lz.copy())

    # All ones with one flip
    lo = np.ones(n, dtype=np.int32)
    lo[0] = 0
    initializations.append(lo.copy())

    # Process each initialization
    for i, init_labels in enumerate(initializations):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 2.0:
            break

        # Allocate time: local search gets 70%, SA gets 30%
        ls_deadline = start_time + (deadline - start_time) * 0.7 + 5
        ls_deadline = min(ls_deadline, deadline - remaining * 0.3)

        labels, cut = batched_local_search(init_labels, ls_deadline)

        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

        # If we still have time, run simulated annealing from the local optimum
        sa_deadline = deadline - 3.0
        if time.time() < sa_deadline - 5:
            sa_labels, sa_cut = simulated_annealing(labels, sa_deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
