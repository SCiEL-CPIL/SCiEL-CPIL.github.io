"""Rewritten program: dynamic local-search independent set with priority assignment."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def _greedy_min_degree(adjacency, n):
    """Greedy min-degree independent set construction."""
    if n == 0:
        return set()
    
    # Use a priority queue of (degree, vertex)
    alive = np.ones(n, dtype=bool)
    degrees = adjacency.sum(axis=1).copy()
    
    # Min-heap of (degree, vertex)
    heap = [(degrees[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    # Track current degree in heap (lazy deletion)
    current_deg = degrees.copy()
    
    independent_set = set()
    
    while heap:
        deg, v = heapq.heappop(heap)
        
        if not alive[v] or deg != current_deg[v]:
            continue
        
        # Select v
        independent_set.add(v)
        alive[v] = False
        
        # Update neighbors
        neighbors = np.where(adjacency[v] & alive)[0]
        for nb in neighbors:
            current_deg[nb] -= 1
            alive[nb] = False  # Mark as not available (neighbor of selected)
            # Push updated degree for neighbor (it won't be selected, but keep consistent)
            # Actually, we can just mark them as dead
    
    return independent_set


def _local_search_improve(adjacency, n, initial_set, max_iter=200):
    """Improve independent set via swap operations."""
    if n == 0:
        return initial_set
    
    # Build adjacency sets for fast lookup
    adj_sets = [set(np.where(adjacency[i])[0]) for i in range(n)]
    
    current_set = set(initial_set)
    in_set = np.zeros(n, dtype=bool)
    for v in current_set:
        in_set[v] = True
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    for _ in range(max_iter):
        improved = False
        
        # Try removing one vertex from the set
        for v in list(current_set):
            # Try adding neighbors of v that are not in the set
            candidates = []
            for nb in adj_sets[v]:
                if not in_set[nb]:
                    candidates.append(nb)
            
            # Try single swaps: remove v, add one candidate
            for c in candidates:
                # Check if c is independent with current_set - {v}
                valid = True
                for u in current_set:
                    if u != v and adjacency[c, u]:
                        valid = False
                        break
                if valid:
                    new_set = (current_set - {v}) | {c}
                    if len(new_set) > best_size:
                        current_set = new_set
                        in_set[v] = False
                        in_set[c] = True
                        best_set = set(current_set)
                        best_size = len(current_set)
                        improved = True
                        break
            
            if improved:
                break
        
        # Try adding a vertex not in the set
        if not improved:
            for v in range(n):
                if in_set[v]:
                    continue
                # Check if v can be added
                valid = True
                for u in current_set:
                    if adjacency[v, u]:
                        valid = False
                        break
                if valid:
                    current_set.add(v)
                    in_set[v] = True
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_set


def _assign_priority(adjacency, n, independent_set):
    """Assign higher priority to vertices in the independent set, lower to others."""
    scores = np.zeros(n, dtype=float)
    
    # Vertices in the independent set get high scores
    for v in independent_set:
        scores[v] = 1.0
    
    # Vertices not in the set get lower scores based on degree
    # Lower degree is better (more likely to be addable)
    degrees = adjacency.sum(axis=1).astype(float)
    for v in range(n):
        if v not in independent_set:
            scores[v] = -degrees[v] / max(1, n)
    
    return scores


def priority(adjacency):
    """Compute priority scores for vertices to maximize independent set size."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit with safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to boolean numpy array if needed
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Step 1: Greedy min-degree construction
    initial_set = _greedy_min_degree(adjacency, n)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return _assign_priority(adjacency, n, initial_set)
    
    # Step 2: Local search improvement
    improved_set = _local_search_improve(adjacency, n, initial_set, max_iter=100)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return _assign_priority(adjacency, n, improved_set)
    
    # Step 3: Try multiple random restarts if time permits
    best_set = improved_set
    best_size = len(improved_set)
    
    remaining_time = deadline - time.time() - 2  # 2s safety margin
    if remaining_time > 5:
        # Try a few random restarts
        rng = np.random.default_rng(42)
        num_restarts = min(10, int(remaining_time / 0.5))
        
        for _ in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            # Random perturbation: remove a random subset, then re-greedy
            if len(best_set) > 0:
                # Remove 1-3 random vertices from the best set
                remove_count = min(3, max(1, len(best_set) // 10))
                to_remove = rng.choice(list(best_set), size=remove_count, replace=False)
                perturbed = set(best_set) - set(to_remove.tolist())
            else:
                perturbed = set()
            
            # Complete with greedy min-degree on remaining vertices
            # Create a modified adjacency where vertices in perturbed are "pre-selected"
            alive = np.ones(n, dtype=bool)
            for v in perturbed:
                alive[v] = False
                # Remove neighbors from consideration
                neighbors = np.where(adjacency[v] & alive)[0]
                alive[neighbors] = False
            
            # Greedy on remaining
            current_deg = (adjacency & alive[:, None] & alive[None, :]).sum(axis=1)
            heap = [(current_deg[i], i) for i in range(n) if alive[i]]
            heapq.heapify(heap)
            
            new_set = set(perturbed)
            while heap:
                deg, v = heapq.heappop(heap)
                if not alive[v] or deg != current_deg[v]:
                    continue
                new_set.add(v)
                alive[v] = False
                neighbors = np.where(adjacency[v] & alive)[0]
                for nb in neighbors:
                    current_deg[nb] -= 1
                    alive[nb] = False
            
            if len(new_set) > best_size:
                best_set = new_set
                best_size = len(new_set)
    
    return _assign_priority(adjacency, n, best_set)
# EVOLVE-BLOCK-END
