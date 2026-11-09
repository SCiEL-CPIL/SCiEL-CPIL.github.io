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
        same = np.sum(adj[i] & (labels == labels[i]))
        opp = np.sum(adj[i] & (labels != labels[i]))
        gains[i] = same - opp
    return gains


def _walksat_step(adj, labels, gains, p=0.1):
    """
    Perform one WalkSAT step:
    - If any flip improves the cut, do it deterministically.
    - Otherwise, with probability p, pick a random vertex to flip (escape move).
    - Otherwise, pick the best non-improving or slightly-worsening flip.
    """
    n = len(labels)
    
    # Find best improving move
    best_gain = np.max(gains)
    
    if best_gain > 0:
        # Deterministic: flip the best improving vertex
        i = np.argmax(gains)
        return i, True  # deterministic move
    
    # No improving move exists
    # With probability p, make a random move (WalkSAT escape)
    if np.random.random() < p:
        i = np.random.randint(n)
        return i, False  # random escape move
    
    # Otherwise, pick the "least bad" move: maximize gain (which is <= 0)
    # This is still a local search step but allows worsening moves
    i = np.argmax(gains)
    return i, False  # non-improving local move


def _local_search_walksat(adjacency, labels, max_time=None, p=0.1):
    """Local search with WalkSAT-style stochastic moves."""
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    gains = _compute_gains(adj, labels)
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    stagnation = 0
    max_stagnation = 5000
    
    while True:
        if time.time() > deadline:
            break
        
        # Perform a WalkSAT step
        i, was_improving = _walksat_step(adj, labels, gains, p)
        
        # Apply the flip
        old_label = labels[i]
        labels[i] = 1 - old_label
        current_cut += gains[i]
        
        # Update gains incrementally
        gains[i] = -gains[i]
        neighbors = np.where(adj[i] == 1)[0]
        for u in neighbors:
            if labels[u] == labels[i]:
                gains[u] += 2
            else:
                gains[u] -= 2
        
        # Track best solution
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            stagnation = 0
        else:
            stagnation += 1
        
        # If too much stagnation, restart with a small random perturbation
        if stagnation > max_stagnation:
            # Perturb a few random vertices
            for _ in range(3):
                v = np.random.randint(n)
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gains[v]
                gains[v] = -gains[v]
                neighbors = np.where(adj[v] == 1)[0]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        gains[u] += 2
                    else:
                        gains[u] -= 2
            stagnation = 0
    
    return best_labels, best_cut


def partition(adjacency):
    """Find a good partition using multiple restarts and WalkSAT local search."""
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
        
        # Run WalkSAT local search
        remaining_time = total_deadline - time.time() - 1
        if remaining_time <= 0:
            break
        
        labels, cut = _local_search_walksat(adjacency, labels, max_time=remaining_time, p=0.1)
        
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
