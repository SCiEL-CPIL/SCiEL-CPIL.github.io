"""Maximum Cut via Spectral Initialization, Fast Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral (Fiedler vector) initialization, fast local search with heap,
    and simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=int)

    # Precompute degrees and neighbor lists
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]
    total_edges = int(adj.sum()) // 2

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def compute_same(labels):
        """Compute same[v] = number of neighbors u where labels[u] == labels[v]."""
        same = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if labels[v] == 0:
                same[v] = int(adj[v, labels == 0].sum())
            else:
                same[v] = int(adj[v, labels == 1].sum())
        return same

    def local_search(labels, time_limit):
        """
        Run local search until no improvement or time limit reached.
        Uses efficient O(degree) updates instead of O(n) row sums.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)

        # Heap of (-gain, vertex)
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            while heap:
                neg_gain, v = heapq.heappop(heap)
                gain = -neg_gain

                # Verify gain is still valid
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Flip vertex v
                old_label = labels[v]
                labels[v] = 1 - old_label
                new_label = labels[v]

                # Update same[v]: when v flips, its same count changes by
                # (neighbors in old group) - (neighbors in new group) = 2*same[v] - deg[v]
                # Actually: new_same[v] = deg[v] - same[v] (since it now counts neighbors in the OTHER group)
                same[v] = degrees[v] - same[v]

                # Update neighbors of v: for each neighbor u,
                # if u was in old_label group, it loses one same neighbor (v left that group)
                # if u was in new_label group, it gains one same neighbor (v joined that group)
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == new_label:
                        # u is now in same group as v, so u gains one same neighbor
                        same[u] += 1
                    else:
                        # u is in the other group, so u loses one same neighbor
                        same[u] -= 1

                    # Check if u now has positive gain
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0:
                        heapq.heappush(heap, (-u_gain, u))

                cut += current_gain
                improved = True
                break

        return labels, cut

    def simulated_annealing(labels, time_limit):
        """
        Run simulated annealing to escape local minima.
        Uses efficient O(degree) updates.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut

        T_start = 10.0
        T_end = 0.05
        cooling_rate = 0.98
        T = T_start

        same = compute_same(labels)

        iterations = 0
        while T > T_end and time.time() < time_limit:
            v = np.random.randint(0, n)

            gain = 2 * same[v] - degrees[v]

            # Accept if gain > 0 or with probability exp(-gain/T)
            if gain > 0 or (np.random.rand() < np.exp(-gain / T)):
                old_label = labels[v]
                labels[v] = 1 - old_label
                new_label = labels[v]

                # Update same[v]: new_same = deg - old_same
                same[v] = degrees[v] - same[v]

                # Update neighbors
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == new_label:
                        same[u] += 1
                    else:
                        same[u] -= 1

                cut += gain

                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()

            T *= cooling_rate
            iterations += 1
            # Occasionally do a local search polish
            if iterations % 500 == 0:
                labels, cut = local_search(labels, time.time() + 0.5)
                same = compute_same(labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()

        return best_labels, best_cut

    def spectral_init():
        """
        Generate initial labeling using the Fiedler vector (second smallest eigenvector).
        """
        try:
            adj_sym = (adj + adj.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(adj_sym)

            # Use the Fiedler vector (index 1, second smallest eigenvalue)
            fiedler = eigvecs[:, 1]

            # Round based on sign
            labels = (fiedler > 0).astype(int)

            # Ensure both classes are non-empty
            if np.all(labels == 0) or np.all(labels == 1):
                labels = (fiedler > np.median(fiedler)).astype(int)

            return labels
        except Exception:
            return None

    rng = np.random.default_rng(42)

    best_labels = None
    best_cut = -1

    # Generate multiple candidate initializations
    candidates = []

    # 1. Spectral (Fiedler) initialization
    spectral = spectral_init()
    if spectral is not None:
        candidates.append(spectral)

    # 2. Random initialization
    random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
    candidates.append(random_labels)

    # 3. Greedy initialization
    greedy_labels = np.zeros(n, dtype=np.int8)
    order = np.argsort(-degrees)
    for v in order:
        if len(neighbors[v]) == 0:
            continue
        count_1 = np.sum(greedy_labels[neighbors[v]] == 1)
        count_0 = np.sum(greedy_labels[neighbors[v]] == 0)
        if count_1 > count_0:
            greedy_labels[v] = 1
        else:
            greedy_labels[v] = 0
    candidates.append(greedy_labels)

    # 4. Additional spectral-based candidates using different eigenvectors
    try:
        adj_sym = (adj + adj.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(adj_sym)

        # Try a few different eigenvectors
        for idx in [0, 2, 3]:
            if idx < n:
                vec = eigvecs[:, idx]
                labels = (vec > np.median(vec)).astype(int)
                if not (np.all(labels == 0) or np.all(labels == 1)):
                    candidates.append(labels)
    except Exception:
        pass

    # 5. Add a few more random candidates
    for _ in range(3):
        random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
        candidates.append(random_labels)

    # Refine each candidate with local search
    for labels in candidates:
        if time.time() > deadline - 15:
            break

        ls_labels, ls_cut = local_search(labels, deadline - 15)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()

    # Run simulated annealing on the best solution found
    if best_labels is not None and time.time() < deadline - 10:
        sa_labels, sa_cut = simulated_annealing(best_labels, deadline - 10)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    # Final local search polish
    if best_labels is not None:
        final_labels, final_cut = local_search(best_labels, deadline - 1)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    # Pairwise swap polish: try swapping pairs of vertices to escape local minima
    if best_labels is not None and time.time() < deadline - 0.5:
        labels = best_labels.copy()
        cut = best_cut
        same = compute_same(labels)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        continue  # Swapping same-label vertices has no effect
                    # Compute gain of swapping i and j
                    # Gain = change in cut size
                    # When we swap i and j:
                    # - i moves from group A to B, j moves from group B to A
                    # Edges affected: (i,u) for all u, (j,u) for all u, and (i,j) if it exists
                    # Net gain = 2*(same[i] + same[j]) - degrees[i] - degrees[j] 
                    #           - 2*adj[i,j] (if i,j edge, it was cut or not, now changes)
                    # Actually let's compute more carefully:
                    # Before: i in group 0, j in group 1 (wlog)
                    # After: i in group 1, j in group 0
                    # 
                    # For edge (i,j): before it was cut (0-1), after it's still cut (1-0). No change.
                    # For edge (i,u), u != j: 
                    #   If u in group 0: before same (0-0), after cut (1-0). Change: -1
                    #   If u in group 1: before cut (0-1), after same (1-1). Change: +1
                    #   Net for i: (count of u in group 1) - (count of u in group 0) = deg[i] - 2*same[i]
                    # For edge (j,u), u != i:
                    #   If u in group 0: before cut (1-0), after same (0-0). Change: +1
                    #   If u in group 1: before same (1-1), after cut (0-1). Change: -1
                    #   Net for j: (count of u in group 0) - (count of u in group 1) = 2*same[j] - deg[j]
                    # Total gain = (deg[i] - 2*same[i]) + (2*same[j] - deg[j])
                    #           = deg[i] - deg[j] - 2*same[i] + 2*same[j]
                    # But wait, we need to be careful about the edge (i,j) being counted in both.
                    # adj[i,j] is counted in deg[i] and deg[j], and in same[i] if they were same group.
                    # They're in different groups, so adj[i,j] is NOT in same[i] or same[j].
                    # The edge (i,j) contribution: before cut, after cut. No change. Good.
                    # But deg[i] includes the edge to j, and deg[j] includes edge to i.
                    # Let me recompute:
                    # gain_i = (neighbors of i in group 1) - (neighbors of i in group 0)
                    #         = (deg[i] - same[i] - adj[i,j]) - same[i]  [excluding j since j is in group 1]
                    # Wait, this is getting complicated. Let me just compute directly.
                    
                    # Direct computation:
                    # gain = 0
                    # For each neighbor u of i (u != j):
                    #   if labels[u] == labels[i]: gain -= 1  (was same, now cut)
                    #   else: gain += 1  (was cut, now same)
                    # For each neighbor u of j (u != i):
                    #   if labels[u] == labels[j]: gain -= 1  (was same, now cut)  
                    #   else: gain += 1  (was cut, now same)
                    # Edge (i,j): no change (was cut, still cut)
                    
                    # Using same[]:
                    # same[i] = count of neighbors of i in same group as i
                    # Neighbors of i in group 1 (other group) = deg[i] - same[i] - adj[i,j]
                    # (subtract adj[i,j] because j is in group 1 but we exclude j)
                    # Neighbors of i in group 0 (same group) = same[i]
                    # Contribution from i: (deg[i] - same[i] - adj[i,j]) - same[i] = deg[i] - 2*same[i] - adj[i,j]
                    
                    # same[j] = count of neighbors of j in same group as j (group 1)
                    # Neighbors of j in group 0 (other group) = deg[j] - same[j] - adj[i,j]
                    # Neighbors of j in group 1 (same group) = same[j]
                    # Contribution from j: (deg[j] - same[j] - adj[i,j]) - same[j] = deg[j] - 2*same[j] - adj[i,j]
                    
                    # Total gain = deg[i] - 2*same[i] - adj[i,j] + deg[j] - 2*same[j] - adj[i,j]
                    #           = deg[i] + deg[j] - 2*same[i] - 2*same[j] - 2*adj[i,j]
                    
                    ai_j = int(adj[i, j])
                    swap_gain = degrees[i] + degrees[j] - 2 * same[i] - 2 * same[j] - 2 * ai_j
                    
                    if swap_gain > 0:
                        # Perform swap
                        labels[i], labels[j] = labels[j], labels[i]
                        
                        # Update same for i and j
                        # i is now in group 1 (was 0), j is now in group 0 (was 1)
                        # Recompute same[i] and same[j]
                        same[i] = int(adj[i, labels == labels[i]].sum())
                        same[j] = int(adj[j, labels == labels[j]].sum())
                        
                        # Update neighbors
                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:
                                same[u] += 1
                            else:
                                same[u] -= 1
                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:
                                same[u] += 1
                            else:
                                same[u] -= 1
                        
                        cut += swap_gain
                        improved = True
                        break  # Restart scan after a swap
            
            if improved:
                best_cut = cut
                best_labels = labels.copy()

    if best_labels is None:
        return np.zeros(n, dtype=int)

    return best_labels.astype(int)

# EVOLVE-BLOCK-END
