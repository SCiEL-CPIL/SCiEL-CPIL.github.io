"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def partition(adjacency):
    """
    Solve Maximum Cut using Optimized Simulated Annealing.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # Precompute neighbors for each vertex to speed up delta updates
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    # Initial random assignment
    labels = np.random.randint(0, 2, n).astype(np.int8)
    
    # Calculate initial cut size
    # cut = total_edges - sum_same_side_edges
    total_edges = np.sum(degrees) / 2.0
    
    # Calculate initial delta (gain of flipping each vertex)
    # delta[v] = 2 * s_v - degrees[v]
    # s_v = sum of edges from v to vertices with same label as v
    delta = np.zeros(n, dtype=np.int32)
    
    for v in range(n):
        s_v = np.dot(adj[v], (labels == labels[v]))
        delta[v] = 2 * s_v - degrees[v]
        
    # Initial cut
    # sum of delta? No.
    # cut = sum of edges crossing.
    # We can compute it once.
    mask0 = (labels == 0)
    mask1 = (labels == 1)
    s0 = np.sum(adj[np.ix_(mask0, mask0)]) / 2.0
    s1 = np.sum(adj[np.ix_(mask1, mask1)]) / 2.0
    current_cut = total_edges - s0 - s1
    
    best_cut = current_cut
    best_labels = labels.copy()
    
    start_time = time.time()
    timeout = 115.0
    
    T_start = 10.0
    T_end = 1e-4
    alpha = 0.9999 # Cooling factor
    T = T_start
    
    # To allow restarts if stuck?
    # Let's just run one long SA.
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Pick random vertex
        v = np.random.randint(n)
        
        # Get gain
        gain = delta[v]
        
        if gain >= 0:
            # Accept
            labels[v] = 1 - labels[v]
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            
            # Update deltas for neighbors of v
            # For each neighbor u of v:
            # If labels[u] == labels[v] (NEW label), then they are now same side.
            # Before flip, labels[u] != labels[v] (OLD label).
            # So edge (u,v) moved from crossing to non-crossing.
            # s_u decreases by 1?
            # Let's re-derive.
            # Let L_v be old label. L_v' be new label.
            # For neighbor u:
            # If L_u == L_v (old):
            #   Before: same side. s_u included edge.
            #   After: L_u != L_v'. Opposite side. s_u excludes edge.
            #   s_u changes by -1.
            #   delta_u changes by 2 * (-1) = -2.
            # If L_u != L_v (old):
            #   Before: opposite side. s_u excluded edge.
            #   After: L_u == L_v'. Same side. s_u includes edge.
            #   s_u changes by +1.
            #   delta_u changes by 2 * (+1) = +2.
            
            # So, for each neighbor u of v:
            # if labels[u] == labels[v] (NEW): delta[u] -= 2
            # else: delta[u] += 2
            
            # Note: labels[v] is already updated.
            new_label_v = labels[v]
            
            for u in neighbors[v]:
                if labels[u] == new_label_v:
                    delta[u] -= 2
                else:
                    delta[u] += 2
                    
            # Also, delta[v] itself changes?
            # delta[v] = 2 * s_v - degrees[v]
            # s_v changes.
            # Before flip, s_v was sum of edges to same side.
            # After flip, s_v' = degrees[v] - s_v.
            # So delta[v] becomes 2 * (degrees[v] - s_v) - degrees[v] = 2*degrees[v] - 2*s_v - degrees[v] = degrees[v] - 2*s_v = - (2*s_v - degrees[v]) = -delta[v].
            # So delta[v] = -delta[v].
            delta[v] = -delta[v]
            
        else:
            # Reject or accept with probability
            if T > 0:
                prob = math.exp(gain / T)
                if np.random.rand() < prob:
                    # Accept
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                    
                    new_label_v = labels[v]
                    for u in neighbors[v]:
                        if labels[u] == new_label_v:
                            delta[u] -= 2
                        else:
                            delta[u] += 2
                    delta[v] = -delta[v]
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    return best_labels
# EVOLVE-BLOCK-END
