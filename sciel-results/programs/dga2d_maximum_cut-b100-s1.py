"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Greedy local improvement with multiple random restarts and time guard.
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Precompute degree for efficiency
    degrees = adjacency.sum(axis=1)
    
    # Try multiple random restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random initial assignment
        labels = np.random.randint(0, 2, size=n)
        
        # Compute initial cut size
        # Cut = sum over edges where labels differ
        # For each vertex v, contribution to cut from v's perspective:
        # edges from v to nodes with different label
        # Total cut = 0.5 * sum_v (edges from v to opposite side)
        
        # Efficient computation: for each v, count neighbors with same label
        # same_count[v] = sum of adjacency[v, labels == labels[v]]
        # cut_size = (sum_v degrees[v] - sum_v same_count[v]) / 2
        
        def compute_cut(labels):
            # For each vertex, count neighbors with same label
            same_counts = np.array([adjacency[v][labels == labels[v]].sum() for v in range(n)])
            total_degree = degrees.sum()
            return (total_degree - same_counts.sum()) / 2.0
        
        cut_size = compute_cut(labels)
        
        # Local improvement: flip vertices that improve the cut
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                if time.time() >= deadline:
                    break
                same = adjacency[v][labels == labels[v]].sum()
                other = adjacency[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
                    # Update cut size incrementally
                    # When we flip v, the change in cut is:
                    # Before: v contributes 'other' to cut (edges to opposite side)
                    # After: v contributes 'same' to cut (edges to what was same side, now opposite)
                    # So delta = same - other
                    cut_size += (same - other)
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
        
        # If we've done a reasonable number of restarts, stop
        if num_restarts >= 10:
            break
    
    if best_labels is None:
        # Fallback: alternating start
        labels = np.arange(n) % 2
        return labels
    
    return best_labels
# EVOLVE-BLOCK-END
