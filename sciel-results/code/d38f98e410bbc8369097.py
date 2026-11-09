import numpy as np
import heapq

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to list of lists for efficient neighbor iteration if needed, 
    # but N=200 is small, so matrix operations are fine.
    # However, for peeling, we need to iterate neighbors.
    
    # Initialize
    deg = adjacency.sum(axis=1).astype(int)
    alive = np.ones(n, dtype=bool)
    
    # Min-heap of (degree, vertex)
    heap = [(deg[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    core_numbers = np.zeros(n, dtype=int)
    
    # We need to handle stale entries in the heap.
    # When we pop (d, v), if alive[v] is False, skip.
    # If alive[v] is True, but deg[v] != d, it's stale? 
    # Actually, if we push (new_deg, u) when u's degree changes, 
    # the old entry (old_deg, u) remains.
    # When we pop, we check if it's the current degree.
    # But since we only decrease degrees, if we pop (d, v) and deg[v] < d, it's stale.
    # If deg[v] == d, it's current.
    
    while heap:
        d, v = heapq.heappop(heap)
        if not alive[v]:
            continue
        if deg[v] != d:
            # Stale entry, push back the current one? 
            # Or just ignore? 
            # If we ignore, we might miss it if it's the min?
            # Better to check: if d > deg[v], it's stale. 
            # Since degrees only decrease, d >= deg[v].
            # If d == deg[v], it's valid.
            # If d > deg[v], it's stale. We should push the correct one? 
            # No, the correct one is already in the heap or will be.
            # Actually, just skip if stale.
            continue
            
        # v is the next vertex to remove
        core_numbers[v] = d
        alive[v] = False
        
        # Update neighbors
        # Find neighbors of v
        neighbors = np.where(adjacency[v, :])[0]
        for u in neighbors:
            if alive[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
                
    # The priority should favor vertices with LOW core numbers.
    # So we return -core_numbers.
    # To break ties, we can add a small penalty for high degree.
    # score = -core_numbers - 0.01 * deg_original?
    # Let's stick to -core_numbers first.
    
    return -core_numbers.astype(float)
