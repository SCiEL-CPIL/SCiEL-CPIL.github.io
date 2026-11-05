import numpy as np
import time

def partition(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
    n = adjacency.shape[0]
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    def cut_size(labels):
        # Number of edges crossing the partition
        # (labels[:, None] != labels[None, :]) creates a bool matrix
        # Multiply by adjacency and sum
        return int(np.sum(adjacency * (labels[:, None] != labels[None, :])))
    
    def improve(labels):
        """
        Local search: Flip the vertex that gives the largest increase in cut size.
        """
        current_labels = labels.copy()
        
        # Limit iterations to prevent infinite loops in pathological cases, 
        # though MaxCut local search usually converges quickly.
        max_iters = 1000 
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            # Compute gains
            mask_0 = (current_labels == 0).astype(np.int8)
            mask_1 = (current_labels == 1).astype(np.int8)
            
            # count_0[v] = number of neighbors of v with label 0
            # count_1[v] = number of neighbors of v with label 1
            count_0 = adjacency @ mask_0
            count_1 = adjacency @ mask_1
            
            # If v is 0: gain = count_1[v] - count_0[v]
            # If v is 1: gain = count_0[v] - count_1[v]
            gains = np.where(current_labels == 0, count_1 - count_0, count_0 - count_1)
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break # Local optimum
            
            # Pick the vertex with max gain
            # If ties, pick randomly among them to add stochasticity
            best_indices = np.where(gains == max_gain)[0]
            v = best_indices[np.random.randint(len(best_indices))]
            
            current_labels[v] = 1 - current_labels[v]
            
        return current_labels

    best_labels = None
    best_cut = -1
    
    total_edges = int(np.sum(adjacency) // 2) # Assuming undirected, symmetric matrix
    
    # Strategy 1: Random Restarts
    # We can do many restarts because N=100 is small.
    # Let's try 100 restarts.
    
    for i in range(200):
        if time.time() > deadline - 10.0:
            break
            
        # Different initializations
        if i == 0:
            labels = np.zeros(n, dtype=np.int8)
        elif i == 1:
            labels = np.ones(n, dtype=np.int8)
        elif i == 2:
            labels = np.arange(n) % 2
        else:
            # Random
            labels = np.random.randint(0, 2, size=n)
            
        # Improve
        improved_labels = improve(labels)
        
        # Evaluate
        c = cut_size(improved_labels)
        
        if c > best_cut:
            best_cut = c
            best_labels = improved_labels.copy()
            
        # Early stop if perfect
        if best_cut == total_edges:
            break
            
    # If we have time, do a few simulated annealing passes on the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        current_labels = best_labels.copy()
        current_cut = best_cut
        T = 5.0
        # Simple SA: pick random vertex, flip if improves or with prob exp(-delta/T)
        # Since we are close to optimum, single flips are cheap.
        
        while time.time() < deadline - 2.0 and T > 0.1:
            v = np.random.randint(n)
            old_label = current_labels[v]
            current_labels[v] = 1 - old_label
            
            new_cut = cut_size(current_labels)
            delta = new_cut - current_cut
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_cut = new_cut
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = current_labels.copy()
            else:
                current_labels[v] = old_label
            
            T *= 0.99 # Slow cooling
            
            # Periodically polish
            if int(T * 100) % 10 == 0: # Roughly every 10% temperature drop
                polished = improve(current_labels)
                pc = cut_size(polished)
                if pc > current_cut:
                    current_labels = polished
                    current_cut = pc
                    if pc > best_cut:
                        best_cut = pc
                        best_labels = polished.copy()

    return best_labels if best_labels is not None else np.zeros(n, dtype=np.int8)
