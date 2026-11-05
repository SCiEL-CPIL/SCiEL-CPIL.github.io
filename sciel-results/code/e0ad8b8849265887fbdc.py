"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    n = adjacency.shape[0]
    
    # Deadline guard: must finish well before 120s wall-clock limit.
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute adjacency as int for faster arithmetic
    adj = adjacency.astype(np.int8)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff))
    
    def improve(labels):
        """Local improvement via single-vertex flips until no improvement."""
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for v in range(n):
                current_label = labels[v]
                same_side = np.sum(adj[v] & (labels == current_label))
                opp_side = np.sum(adj[v] & (labels != current_label))
                gain = same_side - opp_side
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def improve_pair(labels):
        """Local improvement via pair-flips until no improvement."""
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        continue
                    # Gain from flipping both i and j
                    # For vertex i: currently label li. Neighbors on same side as i become opposite, and vice versa.
                    # Flipping i: gain_i = same_i - opp_i (where same_i is count of neighbors of i with label li)
                    # But if we flip both, we need to account for the edge between i and j.
                    # Let's compute the gain more carefully.
                    # Before flip: 
                    #   Edge (i,j) is crossing (labels differ). 
                    #   For i: same_i neighbors (label li) are not crossing, opp_i neighbors (label 1-li) are crossing.
                    #   For j: same_j neighbors (label lj) are not crossing, opp_j neighbors (label 1-lj) are crossing.
                    # After flipping both i and j:
                    #   Edge (i,j) is NOT crossing (labels same).
                    #   For i (new label 1-li): neighbors with label 1-li were opp_i, now same. Neighbors with label li were same_i, now opp.
                    #   For j (new label 1-lj): neighbors with label 1-lj were opp_j, now same. Neighbors with label lj were same_j, now opp.
                    
                    li = labels[i]
                    lj = labels[j]
                    
                    same_i = np.sum(adj[i] & (labels == li))
                    opp_i = np.sum(adj[i] & (labels != li))
                    same_j = np.sum(adj[j] & (labels == lj))
                    opp_j = np.sum(adj[j] & (labels != lj))
                    
                    # The edge (i,j) contributes:
                    # Before: crossing (counted in opp_i and opp_j? No, it's one edge)
                    # Let's think in terms of cut size change.
                    # Cut size change = (new crossings) - (old crossings)
                    # Old crossings involving i or j:
                    #   Edges from i to opposite side: opp_i (includes j if labels differ, which they do)
                    #   Edges from j to opposite side: opp_j (includes i if labels differ, which they do)
                    #   But edge (i,j) is counted once in total cut.
                    # New crossings involving i or j:
                    #   Edges from i (new label) to new opposite side: same_i (neighbors that were same, now opposite)
                    #   Edges from j (new label) to new opposite side: same_j
                    #   Edge (i,j) is no longer crossing.
                    
                    # Gain = (same_i + same_j) - (opp_i + opp_j) 
                    # But we must be careful: the edge (i,j) is in opp_i and in opp_j.
                    # When we compute opp_i, it includes j. When we compute opp_j, it includes i.
                    # The total old crossings involving i or j (excluding double-counting edge (i,j)):
                    #   = opp_i + opp_j - 1 (subtract the double-counted edge (i,j))
                    # The total new crossings involving i or j:
                    #   = same_i + same_j (edge (i,j) is not crossing now, and it's not in same_i or same_j since labels become same)
                    # Wait, after flip, i has label 1-li, j has label 1-lj. Since li != lj, 1-li == lj and 1-lj == li.
                    # So after flip, i and j have the same label.
                    # same_i (before flip) = neighbors of i with label li. This includes j? No, j has label lj != li.
                    # So same_i does NOT include j.
                    # opp_i (before flip) = neighbors of i with label 1-li. This includes j since lj = 1-li.
                    # So opp_i includes j.
                    
                    # Let's use a simpler formula.
                    # gain = 2 * (same_i + same_j - 2) - (opp_i + opp_j - 2) ... this is getting complicated.
                    
                    # Simpler: just compute the cut size before and after for the affected vertices.
                    # Actually, the standard formula for pair flip gain is:
                    # gain = (same_i + same_j) - (opp_i + opp_j)
                    # where same_i, opp_i, same_j, opp_j are computed BEFORE the flip.
                    # Let's verify:
                    # Old cut contribution from edges incident to i or j:
                    #   Sum over neighbors u of i: [label(u) != label(i)]
                    #   Sum over neighbors u of j: [label(u) != label(j)]
                    #   This double-counts edge (i,j) if it exists and is crossing.
                    # New cut contribution:
                    #   Sum over neighbors u of i: [label(u) != 1-label(i)] = [label(u) == label(i)]
                    #   Sum over neighbors u of j: [label(u) != 1-label(j)] = [label(u) == label(j)]
                    #   This does NOT double-count edge (i,j) because after flip, label(i)==label(j), so edge (i,j) is not crossing.
                    
                    # So gain = (same_i + same_j) - (opp_i + opp_j)
                    # But wait, in the "old" sum, edge (i,j) is counted twice (once in opp_i, once in opp_j).
                    # In the "new" sum, edge (i,j) is counted zero times.
                    # So the actual gain should be:
                    # gain = (same_i + same_j) - (opp_i + opp_j) + 2
                    # Because we removed 2 counts of edge (i,j) from the old sum.
                    
                    # Let's test with a simple case: i and j connected, no other edges.
                    # labels: i=0, j=1.
                    # same_i = 0 (no neighbors with label 0 except... j is label 1, so no)
                    # opp_i = 1 (j is label 1)
                    # same_j = 0
                    # opp_j = 1 (i is label 0)
                    # gain = (0+0) - (1+1) + 2 = 0. Correct, flipping both makes it non-crossing, cut goes from 1 to 0. Wait, that's a loss.
                    # Hmm, if we flip both, i becomes 1, j becomes 0. Edge (i,j) is still crossing! 
                    # Oh wait, if li=0, lj=1, then 1-li=1, 1-lj=0. So after flip, i=1, j=0. Still different!
                    # So pair flip only helps if we flip vertices on the SAME side?
                    
                    # Let me reconsider. If labels[i] == labels[j], flipping both might help.
                    # If labels[i] != labels[j], flipping both keeps them different, so the edge (i,j) remains crossing.
                    
                    # So pair flips are only useful for pairs on the same side.
                    if labels[i] == labels[j]:
                        li = labels[i]
                        same_i = np.sum(adj[i] & (labels == li))
                        opp_i = np.sum(adj[i] & (labels != li))
                        same_j = np.sum(adj[j] & (labels == li))
                        opp_j = np.sum(adj[j] & (labels != li))
                        
                        # Before: edge (i,j) is NOT crossing.
                        # After flip: i and j have label 1-li.
                        # New same for i: neighbors with label 1-li = opp_i (but j is now same, so opp_i - 1 if j is neighbor)
                        # This is getting complex. Let's just compute cut size difference directly for the neighborhood.
                        
                        # Simpler approach: just try the flip and compute the gain by checking affected edges.
                        # For now, skip pair flips as they are O(n^2) per iteration and may not be worth it.
                        pass
        
        return labels
    
    # Track best solution found
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initialization strategies
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        if restart == 0:
            labels = np.arange(n) % 2
        elif restart == 1:
            labels = np.ones(n, dtype=int)
        elif restart == 2:
            labels = np.zeros(n, dtype=int)
        else:
            # Random initialization with varying density
            p = 0.5 + 0.2 * np.sin(restart)
            labels = (np.random.random(n) < p).astype(int)
        
        labels = labels.copy()
        
        # Improve via local search
        labels = improve(labels)
        
        # Evaluate
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        # If we found a perfect cut, stop
        total_edges = int(np.sum(np.triu(adj, k=1)))
        if best_cut >= total_edges:
            break
    
    # Final safety check
    if time.time() >= deadline:
        return best_labels if best_labels is not None else np.arange(n) % 2
    
    return best_labels if best_labels is not None else np.arange(n) % 2
# EVOLVE-BLOCK-END
