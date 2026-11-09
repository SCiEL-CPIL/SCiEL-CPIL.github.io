"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def greedy_1flip(labels, max_passes=10):
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def fast_2swap(labels, max_passes=100):
        """Fast 2-swap local search with incremental O(n) updates per accepted swap."""
        labels = labels.copy()
        
        # Initialize sum_same and sum_other
        sum_same = np.zeros(n, dtype=np.int32)
        sum_other = np.zeros(n, dtype=np.int32)
        for v in range(n):
            sum_same[v] = adj[v][labels == labels[v]].sum()
            sum_other[v] = adj[v][labels != labels[v]].sum()
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Shuffle order to avoid bias
            order = list(range(n))
            np.random.shuffle(order)
            for i_idx in range(n):
                if time.time() > deadline:
                    break
                i = order[i_idx]
                li = labels[i]
                for j_idx in range(n):
                    if j_idx == i_idx:
                        continue
                    j = order[j_idx]
                    if labels[i] == labels[j]:
                        continue
                    
                    a = int(adj[i, j])
                    si = int(sum_same[i])
                    oi = int(sum_other[i])
                    sj = int(sum_same[j])
                    oj = int(sum_other[j])
                    
                    gain = si + sj - oi - oj + 2 * a
                    
                    if gain > 0:
                        # Accept swap
                        lj = labels[j]
                        labels[i], labels[j] = lj, li
                        
                        # Update sum_same/sum_other incrementally
                        sum_same[i], sum_other[i] = oi, si
                        sum_same[j], sum_other[j] = oj, sj
                        
                        # Update all other nodes k
                        mask_li = (labels == li)
                        mask_lj = (labels == lj)
                        
                        # Exclude i and j from masks (they already updated)
                        mask_li[i] = False
                        mask_li[j] = False
                        mask_lj[i] = False
                        mask_lj[j] = False
                        
                        delta_li = adj[:, j] - adj[:, i]  # for nodes with label li
                        delta_lj = adj[:, i] - adj[:, j]  # for nodes with label lj
                        
                        sum_same[mask_li] += delta_li[mask_li]
                        sum_same[mask_lj] += delta_lj[mask_lj]
                        sum_other[mask_li] -= delta_li[mask_li]
                        sum_other[mask_lj] -= delta_lj[mask_lj]
                        
                        improved = True
        
        return labels

    def greedy_construct():
        """Greedy construction: assign vertices one by one to maximize crossing edges."""
        labels = np.full(n, -1, dtype=np.int8)
        order = list(range(n))
        np.random.shuffle(order)
        labels[order[0]] = np.random.randint(2)
        for idx in range(1, n):
            if time.time() > deadline:
                break
            v = order[idx]
            cross_if_0 = 0
            cross_if_1 = 0
            for u in range(n):
                if labels[u] == -1 or adj[v, u] == 0:
                    continue
                if labels[u] == 0:
                    cross_if_1 += 1
                else:
                    cross_if_0 += 1
            labels[v] = 0 if cross_if_0 >= cross_if_1 else 1
        for v in range(n):
            if labels[v] == -1:
                labels[v] = np.random.randint(2)
        return labels

    # Strategy 1: Go-with-the-flow + 1-flip + 2-swap
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = greedy_1flip(labels, max_passes=10)
        labels = fast_2swap(labels, max_passes=50)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy 2: Diverse restarts with 1-flip + 2-swap
    for restart in range(1, 61):
        if time.time() > deadline:
            break
        # Diverse initialization strategies
        if restart % 3 == 1:
            labels = np.random.randint(0, 2, size=n)
        elif restart % 3 == 2:
            labels = np.arange(n) % 2
        else:
            labels = greedy_construct()
        
        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final polish with multiple rounds of 1-flip + 2-swap
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        for _ in range(3):
            if time.time() > deadline:
                break
            final_labels = greedy_1flip(final_labels, max_passes=10)
            final_labels = fast_2swap(final_labels, max_passes=50)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
