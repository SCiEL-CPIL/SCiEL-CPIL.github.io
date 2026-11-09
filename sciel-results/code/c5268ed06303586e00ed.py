import numpy as np
import time
import random

def _compute_diffs(adjacency, labels):
    """Compute the number of neighbors with different labels for each vertex."""
    n = len(labels)
    # diff[i] = sum(adj[i] * (labels != labels[i]))
    # We can compute this using matrix multiplication or loops.
    # For N=100, a loop is fine, or vectorized.
    diff = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # neighbors with different label
        # labels != labels[i] is a boolean mask
        diff[i] = np.sum(adjacency[i] & (labels != labels[i]))
    return diff

def _cut_size_from_diffs(diff):
    """Total cut size is sum of diff / 2."""
    return int(np.sum(diff) // 2)

def _simulated_annealing(adjacency, labels, max_time):
    """Run Simulated Annealing starting from given labels."""
    n = len(labels)
    labels = labels.copy()
    diff = _compute_diffs(adjacency, labels)
    cut = _cut_size_from_diffs(diff)
    
    # Precompute neighbors for fast access
    neighbors = [np.where(adjacency[i] == 1)[0] for i in range(n)]
    degrees = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
    
    # Temperature schedule
    # Start high, end low
    T_start = 10.0
    T_end = 0.1
    # Number of steps is adaptive based on time, but we can fix a large number and check time
    # Or just run until time runs out.
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # To make it faster, we can batch some operations or just rely on the fact that N is small.
    # We will run a fixed number of steps per "epoch" and cool down.
    # Or just simple cooling per step.
    
    # Let's use a geometric cooling schedule.
    # We need to estimate how many steps we can do.
    # Let's just do steps and check time every 1000 steps.
    
    current_T = T_start
    # Cooling rate: we want to reach T_end in roughly max_time seconds.
    # Hard to estimate steps per second.
    # Let's use a fixed number of steps, say 10000 steps per restart, and see if it's fast enough.
    # With N=100, 10000 steps should be very fast in Python if optimized.
    
    steps_per_restart = 20000
    cool_rate = (T_end / T_start) ** (1.0 / steps_per_restart)
    
    for step in range(steps_per_restart):
        if time.time() > deadline:
            break
            
        # Pick random vertex
        v = random.randint(0, n - 1)
        
        # Calculate delta
        # delta = degree[v] - 2 * diff[v]
        delta = degrees[v] - 2 * diff[v]
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # Boltzmann factor
            # exp(delta / T)
            # If T is very small and delta is negative, this is 0
            if current_T > 1e-6:
                prob = np.exp(delta / current_T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            # Flip v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            
            # Update cut
            cut += delta
            
            # Update diff[v]
            # diff[v] changes by -delta?
            # delta = same - diff = (deg - diff) - diff = deg - 2*diff
            # new_diff = deg - new_diff?
            # Let's derive:
            # old_diff = D
            # old_same = deg - D
            # new_diff = old_same = deg - D
            # new_same = old_diff = D
            # So new_diff = deg - old_diff
            # diff[v] = degrees[v] - diff[v]
            
            # Update neighbors
            for u in neighbors[v]:
                # If u had same label as v (old), they are now different.
                # If u had different label as v (old), they are now same.
                
                # Check relation BEFORE flip
                if labels[u] == old_label:
                    # They were same, now different
                    # diff[u] increases by 1
                    diff[u] += 1
                    # For v, this neighbor was 'same', now 'diff'.
                    # This is already accounted for in diff[v] = deg - diff[v]?
                    # Let's verify diff[v] update.
                    # We update diff[v] at the end or now?
                    # Let's just recalculate diff[v] from scratch? No, that's O(deg).
                    # Let's stick to: diff[v] = degrees[v] - diff[v]
                    # This assumes we don't need to update diff[v] incrementally per neighbor,
                    # just swap the count.
                    
                else:
                    # They were different, now same
                    # diff[u] decreases by 1
                    diff[u] -= 1
            
            # Update diff[v]
            diff[v] = degrees[v] - diff[v]
            
        # Cool down
        current_T *= cool_rate
        
    return labels, cut

def partition(adjacency):
    """Find a good partition using Simulated Annealing with multiple restarts."""
    n = len(adjacency)
    start_time = time.time()
    # Safety margin: 120s limit, leave 5s for overhead
    total_deadline = start_time + 115
    
    best_labels = None
    best_cut = -1
    
    # Convert adjacency to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.bool_)
    
    # Ensure it's symmetric and boolean
    # The problem states it's a bool matrix
    
    # Number of restarts
    # We don't know how long one SA run takes.
    # Let's do 10 restarts, allocating time evenly.
    # Or just loop until deadline.
    
    restarts = 0
    while time.time() < total_deadline - 10:
        restarts += 1
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Allocate time for this restart
        # If we have 100s left, and this is restart 1 of 10, give it 10s?
        # Simpler: give each restart a fixed time slice, e.g., 5 seconds.
        # If we finish early, we can do more.
        time_slice = 5.0
        
        labels, cut = _simulated_annealing(adjacency, labels, time_slice)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # Optional: Local search refinement on the best label found so far?
        # SA usually converges to a local optimum.
        # We can run a quick hill climb on the result.
        
    # Final refinement: Run local search on the best solution found
    if best_labels is not None and time.time() < total_deadline - 2:
        # Simple hill climb
        labels = best_labels.copy()
        diff = _compute_diffs(adjacency, labels)
        cut = _cut_size_from_diffs(diff)
        neighbors = [np.where(adjacency[i] == 1)[0] for i in range(n)]
        degrees = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
        
        improved = True
        while improved and time.time() < total_deadline - 1:
            improved = False
            for v in range(n):
                delta = degrees[v] - 2 * diff[v]
                if delta > 0:
                    # Flip
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    cut += delta
                    
                    for u in neighbors[v]:
                        if labels[u] == old_label:
                            diff[u] += 1
                        else:
                            diff[u] -= 1
                    diff[v] = degrees[v] - diff[v]
                    improved = True
                    if cut > best_cut:
                        best_cut = cut
                        best_labels = labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
