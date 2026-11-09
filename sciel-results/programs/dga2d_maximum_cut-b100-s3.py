"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int((adjacency & diff).sum() // 2)


def _compute_gains(adj, labels):
    """Compute the gain of flipping each vertex."""
    n = len(labels)
    gains = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Count neighbors with same label
        same = np.sum(adj[i] & (labels == labels[i]))
        # Count neighbors with opposite label
        opp = np.sum(adj[i] & (labels != labels[i]))
        gains[i] = same - opp
    return gains


def _local_search(adjacency, labels, max_time=None):
    """Local search with single and double flips."""
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    gains = _compute_gains(adj, labels)
    current_cut = _cut_size(adjacency, labels)
    
    while True:
        if time.time() > deadline:
            break
            
        # Try single flip
        best_i = np.argmax(gains)
        if gains[best_i] > 0:
            # Flip best_i
            v = best_i
            labels[v] = 1 - labels[v]
            current_cut += gains[v]
            
            # Update gains incrementally
            # For v: gain changes sign
            gains[v] = -gains[v]
            
            # For neighbors of v:
            neighbors = np.where(adj[v] == 1)[0]
            for u in neighbors:
                # If u and v had same label before, they are different now.
                # u's same_count decreased by 1, opp_count increased by 1.
                # gain = same - opp. Change = -1 - 1 = -2.
                # If u and v had diff label before, they are same now.
                # u's same_count increased by 1, opp_count decreased by 1.
                # gain change = 1 - (-1) = 2.
                
                # Check previous label of v (before flip)
                # We flipped v, so current label of v is 1 - old_label.
                # labels[u] is current label of u.
                # If labels[u] == labels[v] (current), they are same now.
                # Before flip, they were different.
                # So gain[u] += 2.
                # If labels[u] != labels[v] (current), they are different now.
                # Before flip, they were same.
                # So gain[u] -= 2.
                
                if labels[u] == labels[v]:
                    gains[u] += 2
                else:
                    gains[u] -= 2
            continue
            
        # Try double flip
        best_delta = 0
        best_i, best_j = -1, -1
        # Iterate all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == 1:
                    if labels[i] == labels[j]:
                        delta = gains[i] + gains[j] - 2
                    else:
                        delta = gains[i] + gains[j] + 2
                else:
                    delta = gains[i] + gains[j]
                
                if delta > best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_delta > 0:
            # Flip best_i and best_j
            v1, v2 = best_i, best_j
            labels[v1] = 1 - labels[v1]
            labels[v2] = 1 - labels[v2]
            current_cut += best_delta
            
            # Recompute gains fully for simplicity and correctness
            gains = _compute_gains(adj, labels)
            continue
            
        # No improvement
        break
        
    return labels, current_cut


def partition(adjacency):
    """Find a good partition using multiple restarts and local search."""
    n = len(adjacency)
    start_time = time.time()
    total_deadline = start_time + 115  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts to try
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > total_deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        labels, cut = _local_search(adjacency, labels, max_time=total_deadline - time.time() - 1)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used up most of our time, stop
        if time.time() > total_deadline - 5:
            break
    
    # If no best found, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
