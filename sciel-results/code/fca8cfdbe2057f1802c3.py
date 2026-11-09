"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition efficiently."""
    diff_matrix = (labels[:, None] != labels[None, :])
    return np.sum(adjacency * diff_matrix)

def _local_improvement(labels, adjacency, deadline):
    """Perform local improvement until no more flips improve the cut."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        # Randomize the order of vertices to avoid bias
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            
            # Count same-side and other-side neighbors
            same = adjacency[v, labels == labels[v]].sum()
            other = adjacency[v, labels != labels[v]].sum()
            
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels

def _second_order_improvement(labels, adjacency, deadline):
    """Try pairs of flips that individually don't help but together do."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        # Sample a subset of vertex pairs to try
        num_pairs = min(500, n * (n - 1) // 2)
        for _ in range(num_pairs):
            if time.time() > deadline:
                break
            
            i, j = np.random.choice(n, 2, replace=False)
            if labels[i] == labels[j]:
                continue
            
            # Compute gain from flipping both i and j
            li = labels[i]
            lj = labels[j]
            
            # Before: i is li, j is lj (li != lj)
            # After: i is 1-li, j is 1-lj (still different)
            
            # Net change in cut from flipping both i and j:
            # For each neighbor k of i (k != j):
            #   if labels[k] == li: edge (i,k) was not crossing, becomes crossing => +1
            #   if labels[k] != li: edge (i,k) was crossing, becomes not crossing => -1
            # For each neighbor k of j (k != i):
            #   if labels[k] == lj: edge (j,k) was not crossing, becomes crossing => +1
            #   if labels[k] != lj: edge (j,k) was crossing, becomes not crossing => -1
            # The edge (i,j): was crossing (li != lj), still crossing (1-li != 1-lj) => 0
            
            # Gain from flipping i (excluding j's contribution which is handled separately)
            gain_i = adjacency[i, labels != li].sum() - adjacency[i, labels == li].sum()
            # Subtract the contribution from edge (i,j) if it exists
            if adjacency[i, j]:
                gain_i -= 1  # This edge was crossing, and after flip it's still crossing, so no change
                # Wait, let me recalculate. 
                # If adjacency[i,j]=1 and li != lj:
                #   Before: edge (i,j) is crossing, contributes +1 to cut
                #   After flip: 1-li != 1-lj, so still crossing, contributes +1
                #   Net change: 0
                # So we shouldn't subtract anything. Let me just compute directly.
            
            # Let's compute the gain more carefully:
            # Gain from flipping i alone (considering all neighbors including j):
            #   neighbors of i with label == li: become +1 each (were not crossing, now crossing)
            #   neighbors of i with label != li: become -1 each (were crossing, now not crossing)
            #   This includes j if adjacency[i,j]=1 and lj != li (which is true)
            #   So gain_i includes the effect on edge (i,j)
            
            # Gain from flipping j alone (considering all neighbors including i):
            #   neighbors of j with label == lj: become +1 each
            #   neighbors of j with label != lj: become -1 each
            #   This includes i if adjacency[i,j]=1 and li != lj (which is true)
            #   So gain_j includes the effect on edge (i,j)
            
            # But when we flip both, the edge (i,j) is counted twice in the gains:
            #   In gain_i: edge (i,j) was crossing (li != lj), after flipping i it becomes not crossing (1-li == lj) => -1
            #   In gain_j: edge (i,j) was crossing (li != lj), after flipping j it becomes not crossing (li == 1-lj) => -1
            #   But actually, after flipping both, edge (i,j) is still crossing (1-li != 1-lj) => +1
            #   So the net change for edge (i,j) is +1 - (-1) - (-1) = +1? No...
            
            # Let me think again. The cut size is the sum of edges crossing.
            # Before: edge (i,j) contributes 1 (since li != lj)
            # After: edge (i,j) contributes 1 (since 1-li != 1-lj)
            # So net change for edge (i,j) is 0.
            
            # The gain from flipping i alone (if we flip only i):
            #   For each neighbor k of i:
            #     if labels[k] == li: +1 (was not crossing, now crossing)
            #     if labels[k] != li: -1 (was crossing, now not crossing)
            #   This includes k=j: since lj != li, contribution is -1
            #   So gain_i = sum_{k in N(i), labels[k]==li} 1 - sum_{k in N(i), labels[k]!=li} 1
            #   = (number of same-side neighbors of i) - (number of other-side neighbors of i)
            #   But wait, that's negative of what I want. Let me redefine.
            
            # Let's define:
            #   same_i = number of neighbors of i with same label as i
            #   other_i = number of neighbors of i with different label from i
            #   gain_i = other_i - same_i  (if positive, flipping i increases cut)
            
            # Similarly for j.
            
            # When we flip both i and j:
            #   The change in cut is:
            #   For neighbors k of i (k != j):
            #     if labels[k] == li: +1
            #     if labels[k] != li: -1
            #   For neighbors k of j (k != i):
            #     if labels[k] == lj: +1
            #     if labels[k] != lj: -1
            #   For edge (i,j): 0 (as shown above)
            
            # So total gain = (other_i - 1 if adjacency[i,j] else 0 - same_i) + (other_j - 1 if adjacency[i,j] else 0 - same_j)
            # Wait, this is getting complicated. Let me just compute it directly.
            
            old_cut = _cut_size(labels, adjacency)
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(labels, adjacency)
            
            if new_cut > old_cut:
                improved = True
            else:
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
    
    return labels

def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        elif restart % 4 == 1:
            # Alternating (from inspiration)
            labels = np.arange(n) % 2
        elif restart % 4 == 2:
            # Random with slight bias
            labels = np.random.randint(0, 2, n)
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            # Start from alternating but with random perturbation
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        # Local improvement
        labels = _local_improvement(labels.copy(), adjacency, deadline)
        
        # Second-order improvement
        labels = _second_order_improvement(labels, adjacency, deadline)
        
        # Compute cut size
        cut = _cut_size(labels, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final check: if we never found a valid result, use a simple default
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
