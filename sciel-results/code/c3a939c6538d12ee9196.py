"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    n = adjacency.shape[0]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    adj = adjacency.astype(np.int8)
    degrees = adj.sum(axis=1)
    total_edges = int(adj.sum()) // 2
    
    def cut_size(labels):
        """Compute cut size efficiently."""
        # Number of edges crossing = (sum over all edges of [label_i != label_j])
        # = (sum of degrees) - 2*(internal edges)
        # Or directly: for each vertex, count neighbors on opposite side, divide by 2
        opp = (adj @ (1 - labels))  # for each vertex, neighbors on opposite side
        return int(np.sum(opp)) // 2
    
    def improve_single(labels, max_passes=200):
        """Fast local search using incremental gain updates."""
        labels = labels.copy()
        # gain[v] = (neighbors on same side) - (neighbors on opposite side)
        # If gain[v] > 0, flipping v increases cut size by gain[v]
        
        # Initialize gain array
        # For vertex v with label l: 
        #   same_side = sum(adj[v] & (labels == l))
        #   opp_side = sum(adj[v] & (labels != l))
        #   gain = same_side - opp_side
        # Note: same_side + opp_side = degree(v)
        # So gain = 2*same_side - degree(v)
        
        same_count = (adj @ labels)  # for each vertex, neighbors with same label as... wait
        # Actually: adj[v] @ labels gives sum of labels of neighbors of v
        # If labels[v] = 0: same_side = count of neighbors with label 0 = adj[v] @ (1 - labels) ... no
        # Let me think again.
        # labels is 0/1. adj[v] is the adjacency row.
        # same_side(v) = number of neighbors u of v such that labels[u] == labels[v]
        # If labels[v] == 0: same_side = sum(adj[v] * (1 - labels))  [neighbors with label 0]
        # If labels[v] == 1: same_side = sum(adj[v] * labels)  [neighbors with label 1]
        # 
        # Actually: same_side(v) = sum over neighbors u of v: [labels[u] == labels[v]]
        # = sum over neighbors u of v: (1 - |labels[u] - labels[v]|)
        # = degree(v) - sum over neighbors u of v: |labels[u] - labels[v]|
        # 
        # And opp_side(v) = sum over neighbors u of v: [labels[u] != labels[v]]
        # = sum over neighbors u of v: |labels[u] - labels[v]|
        # 
        # So gain(v) = same_side - opp_side = degree(v) - 2*opp_side(v)
        # = degree(v) - 2 * (number of neighbors of v on opposite side)
        
        # Compute opp_side for all vertices:
        # opp_side(v) = sum over u: adj[v,u] * |labels[v] - labels[u]|
        # = sum over u: adj[v,u] * (labels[v] + labels[u] - 2*labels[v]*labels[u])
        # = labels[v] * degree(v) + (adj[v] @ labels) - 2 * labels[v] * (adj[v] @ labels)
        # = labels[v] * degree(v) + (1 - 2*labels[v]) * (adj[v] @ labels)
        
        adj_labels = adj @ labels  # vector: for each v, sum of labels of neighbors
        opp_side = labels * degrees + (1 - 2 * labels) * adj_labels
        gain = degrees - 2 * opp_side
        
        for _pass in range(max_passes):
            if time.time() >= deadline - 2.0:
                break
            improved = False
            # Find vertices with positive gain, flip them in order of gain (greedy)
            # For simplicity, iterate and flip if gain > 0
            for v in range(n):
                if gain[v] > 0:
                    # Flip vertex v
                    old_label = labels[v]
                    new_label = 1 - old_label
                    labels[v] = new_label
                    
                    # Update gain for v and its neighbors
                    # When we flip v, for v itself:
                    #   new opp_side(v) = old same_side(v) = degree(v) - old opp_side(v)
                    #   new gain(v) = degree(v) - 2 * new_opp_side(v) = degree(v) - 2*(degree(v) - old_opp_side(v))
                    #              = 2*old_opp_side(v) - degree(v) = -old_gain(v)
                    gain[v] = -gain[v]
                    
                    # For each neighbor u of v:
                    #   The edge (u,v) changes status.
                    #   If labels[u] == old_label (same as v before): 
                    #     Before: edge not crossing. After: edge crossing (since v flipped).
                    #     So opp_side(u) increases by 1, gain(u) decreases by 2.
                    #   If labels[u] != old_label (opposite to v before):
                    #     Before: edge crossing. After: edge not crossing.
                    #     So opp_side(u) decreases by 1, gain(u) increases by 2.
                    
                    neighbors = np.where(adj[v] == 1)[0]
                    for u in neighbors:
                        if labels[u] == old_label:
                            # u was same side as v, now opposite -> opp_side(u) += 1, gain(u) -= 2
                            gain[u] -= 2
                        else:
                            # u was opposite side to v, now same -> opp_side(u) -= 1, gain(u) += 2
                            gain[u] += 2
                    
                    improved = True
            
            if not improved:
                break
        
        return labels
    
    def improve_pair(labels, max_passes=50):
        """Local improvement via pair-flips for pairs on the same side."""
        labels = labels.copy()
        
        for _pass in range(max_passes):
            if time.time() >= deadline - 2.0:
                break
            improved = False
            
            # For each pair (i,j) on the same side, check if flipping both helps
            # This is O(n^2) per pass, so limit passes
            for i in range(n):
                if time.time() >= deadline - 2.0:
                    return labels
                for j in range(i + 1, n):
                    if labels[i] != labels[j]:
                        continue
                    
                    # Both i and j have label li
                    li = labels[i]
                    
                    # Compute gain from flipping both i and j
                    # Before flip:
                    #   For i: same_i = neighbors of i with label li, opp_i = neighbors of i with label 1-li
                    #   For j: same_j = neighbors of j with label li, opp_j = neighbors of j with label 1-li
                    #   Edge (i,j): not crossing (both label li)
                    # 
                    # After flip (both become 1-li):
                    #   For i (now 1-li): new same = old opp_i (but exclude j if j is neighbor... j is now 1-li too, so j is same)
                    #   This is complex. Let's use the formula:
                    #   gain = (same_i + same_j) - (opp_i + opp_j)
                    #   where same_i, opp_i are BEFORE the flip.
                    #   But we need to account for the edge (i,j).
                    #   Before: edge (i,j) not crossing. It's in same_i (j is same side) and same_j (i is same side).
                    #   After: edge (i,j) not crossing (both flipped). 
                    #   The formula (same_i + same_j) - (opp_i + opp_j) should work directly.
                    
                    same_i = int(np.sum(adj[i] & (labels == li)))
                    opp_i = int(np.sum(adj[i] & (labels != li)))
                    same_j = int(np.sum(adj[j] & (labels == li)))
                    opp_j = int(np.sum(adj[j] & (labels != li)))
                    
                    gain = (same_i + same_j) - (opp_i + opp_j)
                    
                    if gain > 0:
                        labels[i] = 1 - li
                        labels[j] = 1 - li
                        improved = True
            
            if not improved:
                break
        
        return labels
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            labels = np.arange(n) % 2
        elif restart == 1:
            labels = np.ones(n, dtype=int)
        elif restart == 2:
            labels = np.zeros(n, dtype=int)
        elif restart == 3:
            # Greedy random: assign each vertex to the side that maximizes immediate cut
            labels = np.zeros(n, dtype=int)
            for v in range(n):
                # Try both labels, pick the one that gives better local cut
                labels_try0 = labels.copy()
                labels_try0[v] = 0
                labels_try1 = labels.copy()
                labels_try1[v] = 1
                cut0 = cut_size(labels_try0)
                cut1 = cut_size(labels_try1)
                labels[v] = 0 if cut0 >= cut1 else 1
        else:
            # Random with varying bias
            p = 0.5 + 0.3 * np.sin(restart * 0.7)
            labels = (np.random.random(n) < p).astype(int)
        
        # Phase 1: Single-vertex local search (fast)
        labels = improve_single(labels)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        # Phase 2: Pair-flip local search (slower, but can escape local optima)
        if time.time() < deadline - 5.0:
            labels2 = improve_pair(labels)
            c2 = cut_size(labels2)
            if c2 > best_cut:
                best_cut = c2
                best_labels = labels2.copy()
            
            # Phase 3: Another round of single-vertex search after pair flips
            if time.time() < deadline - 3.0:
                labels3 = improve_single(labels2, max_passes=100)
                c3 = cut_size(labels3)
                if c3 > best_cut:
                    best_cut = c3
                    best_labels = labels3.copy()
        
        # Check for perfect cut
        if best_cut >= total_edges:
            break
    
    # Final polish: one more careful pass on the best solution
    if best_labels is not None and time.time() < deadline - 2.0:
        final_labels = improve_single(best_labels, max_passes=300)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    return best_labels if best_labels is not None else np.arange(n) % 2
# EVOLVE-BLOCK-END
