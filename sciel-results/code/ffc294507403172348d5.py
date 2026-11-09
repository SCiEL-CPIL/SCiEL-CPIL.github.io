"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0

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
                        # For i: new sum_same = oi (neighbors with label lj, which now includes j correctly)
                        #         new sum_other = si
                        # For j: new sum_same = oj
                        #         new sum_other = sj
                        # For other k:
                        #   If labels[k] == li: sum_same[k] += adj[k,j] - adj[k,i]
                        #   If labels[k] == lj: sum_same[k] += adj[k,i] - adj[k,j]
                        #   sum_other[k] = deg[k] - sum_same[k]
                        
                        # Update i and j
                        sum_same[i], sum_other[i] = oi, si
                        sum_same[j], sum_other[j] = oj, sj
                        
                        # Update all other nodes k
                        # adj[:, i] is column i, adj[:, j] is column j
                        # For k with label li: delta = adj[k,j] - adj[k,i]
                        # For k with label lj: delta = adj[k,i] - adj[k,j]
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
                        
                        # Update sum_other: sum_other[k] = deg[k] - sum_same[k]
                        # But we can also update directly:
                        # For k with label li: sum_other[k] -= delta_li[k]
                        # For k with label lj: sum_other[k] -= delta_lj[k]
                        sum_other[mask_li] -= delta_li[mask_li]
                        sum_other[mask_lj] -= delta_lj[mask_lj]
                        
                        improved = True
        
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

    # Strategy 2: Random restarts with 1-flip + 2-swap
    for restart in range(1, 41):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final polish with more passes
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        final_labels = fast_2swap(final_labels, max_passes=100)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
