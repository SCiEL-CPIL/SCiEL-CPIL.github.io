"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    # Precompute adjacency as int8 for fast matrix-vector products
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)  # int8 array of degrees
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def cut_size(labels):
        # Count edges where labels differ, divide by 2 (each edge counted twice)
        return int((labels[:, None] != labels[None, :]) & adj).sum() // 2
    
    def local_search(labels, start_time):
        """Single-flip local search. Returns improved labels."""
        # Compute gain for each vertex: gain[v] = 2*(edges to opposite set) - degree[v]
        # edges to opposite set = adj @ (1 - labels) if labels is 0/1
        # Actually: for vertex v in set 0, edges to set 1 = (adj @ (1-labels))[v]
        # For vertex v in set 1, edges to set 0 = (adj @ labels)[v]
        # Unified: edges_to_opposite[v] = adj @ (1 - labels) if labels[v]==0, else adj @ labels
        # Simpler: edges_to_opposite = adj @ (1 - 2*labels) ... no
        # Let's use: for each v, num_same = adj[v] @ labels, num_diff = adj[v] @ (1-labels)
        # gain = num_diff - num_same = adj[v] @ (1 - 2*labels)
        
        improved = True
        while improved:
            if time.time() > start_time + 12:
                break
            improved = False
            # Compute gains for all vertices at once
            # For vertex v: if labels[v]==0, gain = (adj @ (1-labels))[v] - (adj @ labels)[v]
            #              if labels[v]==1, gain = (adj @ labels)[v] - (adj @ (1-labels))[v]
            # Unified: gain[v] = (1 - 2*labels[v]) * ((adj @ labels)[v] - (adj @ (1-labels))[v])
            #         = (1 - 2*labels[v]) * (2*(adj @ labels)[v] - degrees[v])
            # Let's verify: if labels[v]=0, gain = 1*(2*adj@labels[v] - deg[v]) = 2*(edges to set1) - deg[v]
            #   edges to set1 = (adj @ labels)[v], edges to set0 = deg[v] - (adj@labels)[v]
            #   gain = edges_to_set1 - edges_to_set0 = (adj@labels)[v] - (deg[v]-(adj@labels)[v]) = 2*(adj@labels)[v]-deg[v] ✓
            # if labels[v]=1, gain = -1*(2*(adj@labels)[v]-deg[v]) = deg[v]-2*(adj@labels)[v]
            #   edges to set0 = deg[v]-(adj@labels)[v], edges to set1 = (adj@labels)[v]
            #   gain = edges_to_set0 - edges_to_set1 = deg[v]-2*(adj@labels)[v] ✓
            
            adj_labels = adj @ labels  # shape (n,), int
            gains = (1 - 2 * labels) * (2 * adj_labels - degrees)
            
            # Find best positive gain vertex
            best_gain = gains.max()
            if best_gain <= 0:
                break
            v = int(np.argmax(gains))
            labels[v] = 1 - labels[v]
            improved = True
        return labels
    
    def simulated_annealing(labels, start_time):
        """Simulated annealing with single-flip moves."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        while T > min_T:
            if time.time() > start_time + 15:
                break
            # Pick random vertex
            v = np.random.randint(n)
            old_label = labels[v]
            labels[v] = 1 - old_label
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_cut = new_cut
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
            else:
                labels[v] = old_label
            T *= cooling
        return best_labels, best_cut
    
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Different initialization strategies
        if r == 0:
            labels = np.arange(n) % 2  # alternating
        elif r == 1:
            labels = np.zeros(n, dtype=np.int8)  # all zeros
        elif r == 2:
            labels = np.ones(n, dtype=np.int8)  # all ones
        elif r == 3:
            labels = rng.randint(0, 2, size=n).astype(np.int8)
        else:
            # Random with bias toward 50/50
            labels = rng.randint(0, 2, size=n).astype(np.int8)
        
        # Local search
        labels = local_search(labels, time.time())
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # Try simulated annealing from this solution if time permits
        if time.time() < deadline - 30:
            sa_labels, sa_cut = simulated_annealing(labels, time.time())
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
                # Do local search from the SA result
                ls_labels = local_search(sa_labels, time.time())
                ls_cut = cut_size(ls_labels)
                if ls_cut > best_cut:
                    best_cut = ls_cut
                    best_labels = ls_labels.copy()
    
    # Final local search pass on best solution
    if best_labels is not None and time.time() < deadline - 5:
        final_labels = local_search(best_labels, time.time())
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_labels = final_labels
            best_cut = final_cut
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
