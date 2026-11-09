"""Maximum Cut via GW-style spectral embedding + fast 1-flip + optimized 2-swap local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def flip_greedy(labels, max_passes=50):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def tabu_search(labels, max_iter=5000):
        """Cerny tabu search: accepts worsening moves with tabu list to escape local optima."""
        labels = labels.copy()
        # Precompute same_count
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        # Tabu list: dict vertex -> iteration when it becomes un-tabu
        tabu_until = {}
        tabu_len = 20
        current_cut = compute_cut(labels)
        best_cut_ts = current_cut
        best_labels_ts = labels.copy()
        iteration = 0
        
        # Precompute degree array for speed
        deg_arr = degrees
        
        while iteration < max_iter:
            if time.time() > deadline:
                break
            
            # Find best move (including worsening, respecting tabu)
            best_gain = -999999
            best_v = -1
            best_ascent = False
            
            # Sample a random subset of vertices to consider (for speed)
            # For n=100, we can afford to check all
            for v in range(n):
                if v in tabu_until and tabu_until[v] > iteration:
                    continue
                deg = deg_arr[v]
                sc = same_count[v]
                gain = 2 * sc - deg  # gain in cut from flipping v
                # Aspiration: allow tabu move if it improves best
                if gain > best_gain or (gain == best_gain and v not in tabu_until):
                    best_gain = gain
                    best_v = v
                    best_ascent = (gain > 0)
            
            if best_v < 0:
                break
            
            # Perform the flip
            v = best_v
            old_label = labels[v]
            labels[v] = 1 - old_label
            tabu_until[v] = iteration + tabu_len
            
            # Update same_count
            old_sc_v = same_count[v]
            same_count[v] = deg_arr[v] - old_sc_v
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            # Update cut
            current_cut += best_gain
            iteration += 1
            
            if current_cut > best_cut_ts:
                best_cut_ts = current_cut
                best_labels_ts = labels.copy()
        
        return best_labels_ts

    def two_swap_search(labels, max_passes=20):
        """Optimized 2-swap local search with precomputed and incrementally updated counts."""
        labels = labels.copy()
        
        # Precompute: for each vertex v, count neighbors in same partition
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                li = labels[i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if labels[i] == labels[j]:
                        continue
                    
                    lj = labels[j]
                    
                    # Gain from swapping i and j:
                    # For vertex i: currently has same_count[i] same-side neighbors.
                    #   After swap, i is on side lj. Its neighbors that were on lj become same-side.
                    #   Neighbors on li (excluding j) become cross-side.
                    #   j is on lj, so if i-j edge exists, j was cross-side before, same-side after.
                    # 
                    # Let's compute:
                    # n_i_to_lj = number of neighbors of i on side lj (excluding j if adjacent)
                    # n_i_to_li = number of neighbors of i on side li (excluding j if adjacent)
                    # same_count[i] = n_i_to_li + (1 if adj[i,j] and lj==li else 0)
                    # Since lj != li, same_count[i] = n_i_to_li
                    # n_i_to_lj = degrees[i] - same_count[i] - (1 if adj[i,j] else 0)
                    # 
                    # After swap, i is on lj:
                    #   new_same_i = n_i_to_lj + (1 if adj[i,j] else 0)  [j is now on li, so j is cross-side]
                    #   Wait, j moves to li, so j is cross-side for i (which is now on lj).
                    #   new_same_i = n_i_to_lj
                    # 
                    # Similarly for j:
                    # same_count[j] = n_j_to_lj
                    # n_j_to_li = degrees[j] - same_count[j] - (1 if adj[i,j] else 0)
                    # After swap, j is on li:
                    #   new_same_j = n_j_to_li
                    #
                    # Change in cut:
                    #   Old: i contributes (degrees[i] - same_count[i]) cross edges
                    #         j contributes (degrees[j] - same_count[j]) cross edges
                    #   New: i contributes (degrees[i] - new_same_i) cross edges
                    #         j contributes (degrees[j] - new_same_j) cross edges
                    #   But we double-count the i-j edge if it exists.
                    #
                    # Simpler: gain = (new_same_i + new_same_j) - (same_count[i] + same_count[j])
                    #   because total cut = total_edges - sum_v same_count[v] / 2... no.
                    #
                    # Actually: cut = sum over edges of (labels differ). 
                    # For vertex v, number of cross edges = degrees[v] - same_count[v].
                    # Total cross edge count (counting each edge once) = 0.5 * sum_v (degrees[v] - same_count[v])
                    # So improving same_count reduces cut.
                    #
                    # Gain in same_count sum = (new_same_i + new_same_j) - (same_count[i] + same_count[j])
                    # If gain > 0, cut decreases... wait, we want to INCREASE cut.
                    #
                    # Cut = 0.5 * sum_v (deg[v] - same_count[v]) = total_edges - 0.5 * sum_v same_count[v]
                    # So to increase cut, we want to DECREASE sum of same_count.
                    #
                    # Let me recompute. For a swap of i (on li) and j (on lj), li != lj:
                    #
                    # For i: 
                    #   Before: same-side neighbors = same_count[i] (all on li)
                    #   After: i is on lj. Same-side neighbors = neighbors of i that are on lj.
                    #     = (degrees[i] - same_count[i] - adj[i,j]) + 0  [j moves away]
                    #     Wait, j was on lj, so j was a same-side neighbor of i? No, i is on li, j on lj, so j is cross-side.
                    #     So same_count[i] counts neighbors on li. j is not counted.
                    #     Neighbors on lj = degrees[i] - same_count[i]
                    #     After swap, i is on lj, so same-side = neighbors on lj = degrees[i] - same_count[i]
                    #     But j moves to li, so j is no longer on lj. If adj[i,j], j was counted in "neighbors on lj" before?
                    #     No, j was on lj, i on li, so j was cross-side for i. j is a neighbor on lj.
                    #     After swap, i on lj, j on li. j is cross-side for i. 
                    #     So same-side for i after = (neighbors on lj) - (1 if adj[i,j] else 0)
                    #     = (degrees[i] - same_count[i]) - adj[i,j]
                    #
                    # For j:
                    #   Before: same-side neighbors = same_count[j] (all on lj)
                    #   After: j is on li. Same-side = neighbors on li.
                    #     Neighbors on li = degrees[j] - same_count[j]
                    #     But i moves to lj, so i is no longer on li. If adj[i,j], i was a neighbor on li?
                    #     No, i was on li, j on lj, so i was cross-side for j. i is a neighbor on li.
                    #     After swap, j on li, i on lj. i is cross-side for j.
                    #     So same-side for j after = (neighbors on li) - (1 if adj[i,j] else 0)
                    #     = (degrees[j] - same_count[j]) - adj[i,j]
                    #
                    # Change in sum of same_count:
                    #   delta_same = [(degrees[i] - same_count[i] - adj[i,j]) + (degrees[j] - same_count[j] - adj[i,j])]
                    #              - [same_count[i] + same_count[j]]
                    #   = degrees[i] + degrees[j] - 2*same_count[i] - 2*same_count[j] - 2*adj[i,j]
                    #
                    # Cut change = -0.5 * delta_same (since cut = total_edges - 0.5*sum_same)
                    # So gain in cut = 0.5 * delta_same
                    # We want gain > 0, i.e., delta_same > 0.
                    
                    e_ij = adj[i, j]
                    delta_same = (degrees[i] - same_count[i] - e_ij) + (degrees[j] - same_count[j] - e_ij) - same_count[i] - same_count[j]
                    # = degrees[i] + degrees[j] - 2*same_count[i] - 2*same_count[j] - 2*e_ij
                    
                    if delta_same > 0:
                        # Perform swap
                        labels[i] = lj
                        labels[j] = li
                        improved = True
                        
                        # Update same_count for i and j and their neighbors
                        old_same_i = same_count[i]
                        old_same_j = same_count[j]
                        same_count[i] = degrees[i] - old_same_i - e_ij
                        same_count[j] = degrees[j] - old_same_j - e_ij
                        
                        # Update neighbors of i
                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:  # labels[i] is now lj
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1
                        
                        # Update neighbors of j
                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:  # labels[j] is now li
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1
            
            if not improved:
                break
        
        return labels

    def gw_spectral_partition():
        """Goemans-Williamson style: full spectral embedding with random hyperplane."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Use all eigenvectors except the first (zero eigenvalue)
            # For GW, we want a rank-(n-1) embedding
            k = n - 1
            embedding = eigenvectors[:, 1:k+1]
            
            # Normalize rows to unit vectors (GW constraint)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
            # Random hyperplane
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def gw_spectral_partition_low_rank(k=16):
        """GW-style with reduced rank for speed, multiple random hyperplanes."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k+1]
            
            # Normalize rows to unit vectors
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
            # Random hyperplane
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    best_cut = -1
    best_labels = None

    # Generate multiple initial partitions
    init_labels_list = []
    
    # 1. Full GW spectral (rank n-1)
    if time.time() < deadline:
        init_labels_list.append(gw_spectral_partition())
    
    # 2. GW spectral with reduced rank, multiple random hyperplanes
    for _ in range(8):
        if time.time() > deadline:
            break
        init_labels_list.append(gw_spectral_partition_low_rank(k=16))
    
    # 3. Fiedler vector (simplest spectral)
    try:
        D = np.diag(degrees)
        L = D - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        init_labels_list.append((fiedler > 0).astype(np.int8))
    except Exception:
        pass
    
    # 4. Alternating pattern
    init_labels_list.append(np.arange(n) % 2)
    
    # 5. Random
    init_labels_list.append(np.random.randint(0, 2, size=n, dtype=np.int8))

    # Process each initial label assignment
    for init_labels in init_labels_list:
        if time.time() > deadline:
            break
        
        # 1-flip greedy
        labels1 = flip_greedy(init_labels, max_passes=40)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()
        
        # 2-swap after 1-flip
        if time.time() < deadline:
            labels2 = two_swap_search(labels1, max_passes=10)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()
        
        # Tabu search after 2-swap to escape local optima
        if time.time() < deadline:
            labels3 = tabu_search(labels2, max_iter=3000)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

    # Additional random restarts with spectral bias
    num_random = 20
    for r in range(num_random):
        if time.time() > deadline:
            break
        
        # Mix: 50% pure random, 50% spectral-based
        if r % 2 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        else:
            labels = gw_spectral_partition_low_rank(k=8)
        
        # 1-flip greedy
        labels = flip_greedy(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # 2-swap
        if time.time() < deadline:
            labels_sw = two_swap_search(labels, max_passes=8)
            cut_sw = compute_cut(labels_sw)
            if cut_sw > best_cut:
                best_cut = cut_sw
                best_labels = labels_sw.copy()
        
        # Tabu search on best from this restart
        if time.time() < deadline:
            start_tabu = labels_sw if cut_sw > cut else labels
            labels_tab = tabu_search(start_tabu, max_iter=2000)
            cut_tab = compute_cut(labels_tab)
            if cut_tab > best_cut:
                best_cut = cut_tab
                best_labels = labels_tab.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
