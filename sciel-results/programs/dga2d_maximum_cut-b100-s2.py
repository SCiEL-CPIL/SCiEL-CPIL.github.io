"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq
from collections import defaultdict


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 150 seconds total, leave 15s safety margin
    start_time = time.time()
    deadline = start_time + 135.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1  # ±1 vector
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def flip_gain(v, labels, adj):
        """Gain in cut size if we flip vertex v."""
        same = adj[v, labels == labels[v]].sum()
        deg = adj[v].sum()
        return 2 * same - deg
    
    def best_first_local_search(labels, adj, time_limit=None):
        """Best-first local search using a priority queue for flip ordering."""
        labels = labels.copy()
        
        # Build initial priority queue with gains for all vertices
        heap = []
        for v in range(n):
            gain = flip_gain(v, labels, adj)
            if gain > 0:
                heapq.heappush(heap, (-gain, v))
        
        # Track which vertices are currently in the heap
        in_heap = set(range(n))
        
        iters = 0
        while heap and iters < 50000:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            neg_gain, v = heapq.heappop(heap)
            gain = -neg_gain
            
            # Check if this vertex is still in the heap (not already processed)
            if v not in in_heap:
                continue
            in_heap.discard(v)
            
            # Recompute gain in case labels changed
            actual_gain = flip_gain(v, labels, adj)
            if actual_gain <= 0:
                continue
            
            # Flip vertex v
            labels[v] = 1 - labels[v]
            
            # Update gains for neighbors of v
            neighbors = np.nonzero(adj[v])[0]
            for u in neighbors:
                if u in in_heap:
                    in_heap.discard(u)
                new_gain = flip_gain(u, labels, adj)
                if new_gain > 0:
                    heapq.heappush(heap, (-new_gain, u))
                    in_heap.add(u)
        
        return labels
    
    def simulated_annealing(labels, adj, iters=100000, t_start=2.0, t_end=0.01, time_limit=None):
        """Simulated annealing for max cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            # Pick random vertex
            v = random.randint(0, n - 1)
            gain = flip_gain(v, labels, adj)
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    def multi_start_search(adj, deadline):
        """Multiple starts with different initializations."""
        best_labels = None
        best_cut = -1
        
        # Start 1: alternating
        labels = np.arange(n) % 2
        labels = best_first_local_search(labels, adj, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        # Start 2-8: random starts
        for start_iter in range(7):
            if time.time() > deadline:
                break
            labels = np.random.randint(0, 2, size=n)
            labels = best_first_local_search(labels, adj, time_limit=deadline)
            c = cut_size(labels)
            if c > best_cut:
                best_cut = c
                best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Phase 1: Multi-start best-first local search
    best_labels, best_cut = multi_start_search(adj, deadline)
    
    # Phase 2: Simulated annealing from best solution
    if time.time() < deadline - 10:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=150000, t_start=2.0, t_end=0.01,
            time_limit=deadline - 10
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: Final best-first local search polish
    if time.time() < deadline:
        best_labels = best_first_local_search(best_labels, adj, time_limit=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
