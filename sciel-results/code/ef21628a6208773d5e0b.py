import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    deg = adj.sum(axis=1)
    
    # Precompute neighbors for fast iteration
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    
    best_labels = np.zeros(n, dtype=np.int8)
    best_cut = -1
    
    # Deadline: 140s to be safe (limit is 150s)
    deadline = time.time() + 140.0
    
    # Helper to calculate cut and k array from labels
    def compute_state(labels):
        cut = 0
        k = np.zeros(n, dtype=np.int32)
        for i in range(n):
            # Count neighbors with same label
            same_mask = labels == labels[i]
            k[i] = adj[i][same_mask].sum()
            # Cut size is sum of edges to different labels / 2
            # Or simply sum of edges to different labels
            diff_mask = labels != labels[i]
            cut += adj[i][diff_mask].sum()
        return cut // 2, k

    # Helper for greedy local search
    def greedy_polish(labels, k, cut):
        improved = True
        while improved:
            improved = False
            for v in range(n):
                delta = 2 * k[v] - deg[v]
                if delta > 0:
                    # Flip v
                    labels[v] = 1 - labels[v]
                    cut += delta
                    old_label_v = 1 - labels[v]
                    k[v] = deg[v] - k[v]
                    for w in neighbors[v]:
                        if labels[w] == old_label_v:
                            k[w] -= 1
                        else:
                            k[w] += 1
                    improved = True
        return cut, k

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        # Initialize random labels
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        cut, k = compute_state(labels)
        
        # 1. Greedy Polish
        cut, k = greedy_polish(labels, k, cut)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # Check time before SA
        if time.time() > deadline - 5:
            break
            
        # 2. Simulated Annealing
        # Temperature schedule
        # Start T high enough to accept bad moves, cool down
        T_start = 10.0
        T_end = 0.1
        # Number of steps for SA. We'll do a time-bounded number of steps.
        # Estimate steps per second: ~100k-500k depending on deg. Let's say 200k.
        # We'll just loop and check time.
        
        T = T_start
        # Cool down factor per step
        # We want to reach T_end after some number of steps.
        # Let's do SA for a fixed fraction of remaining time, e.g., 1 second per restart?
        # Or just run until time runs out for this restart?
        # Better: Run SA for a fixed number of steps, say 5000, then polish again.
        # If time allows, do more rounds.
        
        sa_steps = 10000
        for step in range(sa_steps):
            if time.time() > deadline - 5:
                break
                
            # Cooling: linear or geometric? Geometric is common.
            # T = T_start * (T_end/T_start)^(step/sa_steps)
            # Or simple linear decay for simplicity and speed
            T = T_start - (T_start - T_end) * (step / sa_steps)
            
            v = np.random.randint(n)
            delta = 2 * k[v] - deg[v]
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # Boltzmann factor
                # Avoid overflow/underflow
                if T > 1e-6:
                    prob = np.exp(delta / T)
                else:
                    prob = 0.0
                accept = np.random.random() < prob
                
            if accept:
                # Flip v
                labels[v] = 1 - labels[v]
                cut += delta
                old_label_v = 1 - labels[v]
                k[v] = deg[v] - k[v]
                for w in neighbors[v]:
                    if labels[w] == old_label_v:
                        k[w] -= 1
                    else:
                        k[w] += 1
        
        # 3. Final Polish after SA
        cut, k = greedy_polish(labels, k, cut)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # If time is very low, break
        if time.time() > deadline - 2:
            break

    return best_labels
