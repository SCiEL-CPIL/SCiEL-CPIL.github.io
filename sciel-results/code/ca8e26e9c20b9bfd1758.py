"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # cut = sum over edges (i,j) where labels[i] != labels[j]
    # Using matrix operations: for each vertex, count neighbors in opposite set
    # cut = 0.5 * sum_v deg_v - 2 * (sum of edges within same set)
    # Simpler: use the adjacency matrix and label vectors
    # Let's use a direct approach that's fast for n=100
    n = len(labels)
    # For each vertex, count how many neighbors are in the opposite set
    # labels is 0/1 array
    # opposite = 1 - labels
    # For vertex v, count = adj[v] @ opposite
    opposite = 1 - labels
    counts = adj @ opposite
    return int(np.sum(counts) // 2)


def _flip_gain(adj, labels, v):
    """Compute the gain (change in cut size) if we flip vertex v."""
    # Current: vertex v contributes adj[v] @ (1 - labels[v]) to the cut
    # If we flip v, it contributes adj[v] @ labels (since 1 - (1-labels[v]) = labels[v])
    # But we need to be careful: the edge (v,u) is counted once in the total cut
    # 
    # Let's think differently. The total cut can be written as:
    # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - labels[i]*labels[j] if same... )
    # Actually, a cleaner way:
    # For vertex v, its contribution to the cut is the number of neighbors in the opposite set.
    # If we flip v, its new contribution is the number of neighbors in the SAME set (old).
    # 
    # Let same_count = number of neighbors of v with same label as v
    # Let other_count = number of neighbors of v with different label from v
    # 
    # Current contribution of v to cut: other_count
    # New contribution of v to cut: same_count
    # 
    # But flipping v also affects the contributions of v's neighbors!
    # For each neighbor u of v:
    #   If u was in opposite set to v (so edge (v,u) was in cut), after flip u is in same set as v (edge not in cut)
    #   If u was in same set as v (so edge (v,u) was NOT in cut), after flip u is in opposite set (edge in cut)
    # 
    # So the total change in cut size:
    #   Delta = (new_v_contribution - old_v_contribution) + sum over neighbors u of (change in u's contribution)
    # 
    # For v: delta_v = same_count - other_count
    # For each neighbor u of v:
    #   If labels[u] != labels[v] (u in opposite set): u's contribution decreases by 1 (edge v-u removed from cut)
    #   If labels[u] == labels[v] (u in same set): u's contribution increases by 1 (edge v-u added to cut)
    #   So delta_u = (1 if labels[u]==labels[v] else -1) for each neighbor u
    # 
    # Total delta = (same_count - other_count) + (same_count - other_count)
    #              = 2 * (same_count - other_count)
    # 
    # Wait, that's not right. Let me redo.
    # 
    # Actually, the standard result for max-cut local search:
    # gain(v) = 2 * (sum of adj[v, same_set] - sum of adj[v, opposite_set])
    #         = 2 * (same_count - other_count)
    
    same_mask = (labels == labels[v])
    other_mask = (labels != labels[v])
    same_count = adj[v, same_mask].sum()
    other_count = adj[v, other_mask].sum()
    return 2 * (same_count - other_count)


def _local_search(adj, labels, max_iters=1000):
    """Perform greedy local search, flipping vertices if it improves the cut."""
    n = len(labels)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for v in range(n):
            gain = _flip_gain(adj, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _simulated_annealing(adj, labels, start_temp=10.0, end_temp=0.1, cooling=0.999, max_iters=5000):
    """Perform simulated annealing on the partition."""
    n = len(labels)
    temp = start_temp
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for _ in range(max_iters):
        if temp <= end_temp:
            break
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        old_label = labels[v]
        labels[v] = 1 - old_label
        new_cut = _cut_size(adj, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
        
        temp *= cooling
    
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        elif restart % 3 == 1:
            # Alternating initialization
            labels = np.arange(n) % 2
        else:
            # Random with bias
            labels = (np.random.rand(n) > 0.5).astype(np.int8)
        
        # Local search
        labels = _local_search(adj, labels.copy())
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we have time, try simulated annealing from this solution
        if time.time() < deadline - 5.0:
            sa_labels = _simulated_annealing(adj, labels.copy(), max_iters=2000)
            sa_cut = _cut_size(adj, sa_labels)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    if best_labels is None:
        # Fallback: return alternating
        return np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
