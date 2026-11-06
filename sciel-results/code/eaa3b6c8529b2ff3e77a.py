# EVOLVE-BLOCK-START

import time
import random
import sys

def _has_k33(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains K3,3.
    Returns True if found, False otherwise. Uses bitsets for speed."""
    for a1 in range(n):
        for a2 in range(a1 + 1, n):
            for a3 in range(a2 + 1, n):
                common = adj[a1] & adj[a2] & adj[a3]
                # Exclude a1, a2, a3 themselves
                common &= ~(1 << a1) & ~(1 << a2) & ~(1 << a3)
                # Count bits
                # Python 3.10+ has int.bit_count(), but we stick to bin().count('1') for compatibility or use a lookup if needed.
                # For n=20, bin().count is fast enough.
                if common.bit_count() >= 3:
                    return True
    return False

def _count_edges(adj, n):
    count = 0
    for i in range(n):
        count += adj[i].bit_count()
    return count // 2

def _construct_greedy(n=20, timeout=10, seed=None):
    """
    Construct a dense K3,3-free graph using a greedy approach.
    1. Initialize with a sparse random graph or empty.
    2. Greedily add edges that do not create a K3,3.
    3. Repeat with different random seeds/orderings if time permits.
    """
    start_time = time.time()
    deadline = start_time + timeout
    
    if seed is not None:
        random.seed(seed)
    
    best_edges = 0
    best_adj = None
    
    # We can try starting from different densities
    # Starting from empty is safe but slow to fill.
    # Starting from K_{1, n-1} is a good baseline.
    # Let's try a few random initializations and greedy fills.
    
    # Strategy 1: Start with K_{1, n-1} (star)
    adj_star = [0] * n
    for i in range(1, n):
        adj_star[0] |= (1 << i)
        adj_star[i] |= (1 << 0)
    
    # Strategy 2: Start with empty
    adj_empty = [0] * n
    
    # Strategy 3: Start with random sparse graph (p=0.1)
    adj_rand = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.1:
                adj_rand[i] |= (1 << j)
                adj_rand[j] |= (1 << i)
    # Clean adj_rand first if it has K3,3
    # But for p=0.1, n=20, it's very likely K3,3-free.
    # Let's verify and clean if needed.
    if _has_k33(adj_rand, n):
        # Remove edges until valid
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_rand[i] >> j) & 1:
                    if _has_k33(adj_rand, n):
                        # Try removing this edge
                        adj_rand[i] &= ~(1 << j)
                        adj_rand[j] &= ~(1 << i)
                        if not _has_k33(adj_rand, n):
                            continue
                        else:
                            # Revert if it didn't help? No, if it still has K3,3, we need to remove more.
                            # This simple loop might not clean it perfectly.
                            pass
        # If still invalid, discard this initialization
        if _has_k33(adj_rand, n):
            adj_rand = None

    candidates = [adj_star, adj_empty]
    if adj_rand is not None:
        candidates.append(adj_rand)
        
    # Also add some random dense-ish graphs that are cleaned?
    # Maybe just stick to these 3 types for now to save time.
    
    for init_adj in candidates:
        if init_adj is None:
            continue
        if time.time() > deadline:
            break
            
        adj = init_adj[:] # Copy
        
        # Greedy Add Phase
        # We want to add edges. To make it deterministic per seed but varied,
        # we can shuffle the order of edge attempts.
        
        # Generate list of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        # Pass 1: Add any edge that is safe
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _has_k33(adj, n):
                    pass # Keep it
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Pass 2: Try to add edges again? Sometimes adding an edge opens up others?
        # No, adding edges only creates more constraints. So one pass is sufficient for a given order.
        # However, different orders yield different results.
        
        # Local Search: Try to swap edges?
        # If we remove an edge (a,b), can we add two edges (c,d) and (e,f)?
        # This is complex. Let's just do multiple random restarts of the greedy phase.
        
        current_count = _count_edges(adj, n)
        if current_count > best_edges:
            best_edges = current_count
            best_adj = adj[:]
            
        # If we have time, do a few more restarts with different shuffles
        # The loop above only did one shuffle. We should loop this whole block a few times.
        
    # Actually, the structure above only does 1 shuffle per candidate.
    # Let's restructure to allow multiple shuffles if time permits.
    
    # Revised Loop for Multiple Restarts
    # We will just run the greedy add with random shuffles multiple times.
    
    # Reset best_adj to the best found so far
    # The code above already does this.
    
    # Let's add a second phase: Local search on the best_adj found.
    if best_adj is not None and time.time() < deadline:
        adj = best_adj[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove an edge and see if we can add more?
            # Or just try to add more edges (should be none left).
            # Try "edge swaps": remove (u,v), then try to add (x,y) and (z,w).
            # This is too slow.
            
            # Alternative: Randomly remove an edge, then run greedy add again.
            # This might escape local optima.
            if random.random() < 0.5: # Only do this half the time to save time?
                # Pick a random edge to remove
                edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
                if edges_list:
                    u, v = random.choice(edges_list)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Run greedy add again
                    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((adj[i] >> j) & 1)]
                    random.shuffle(all_edges)
                    for x, y in all_edges:
                        if time.time() > deadline:
                            break
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        if _has_k33(adj, n):
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                    
                    new_count = _count_edges(adj, n)
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        improved = True
                    else:
                        # Revert to best_adj? Or keep current?
                        # If we didn't improve, we might be in a worse spot.
                        # Let's revert to the previous best state to avoid drifting.
                        adj = best_adj[:]
                        
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback to star
        best_adj = [0] * n
        for i in range(1, n):
            best_adj[0] |= (1 << i)
            best_adj[i] |= (1 << 0)
            
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges


def construct():
    """Return a list of edges [u, v] of a K3,3-free graph on 20 vertices, maximizing edges."""
    start_time = time.time()
    deadline = start_time + 120 - 5  # 5s safety margin
    
    best_edges = None
    best_count = 0
    
    # Run multiple restarts
    # Each restart gets a chunk of time
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        seed = i * 12345 + 67890
        try:
            edges = _construct_greedy(20, timeout=time_per_restart, seed=seed)
            count = len(edges)
            
            # Verify validity
            adj = [0] * 20
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            if not _has_k33(adj, 20) and count > best_count:
                best_count = count
                best_edges = edges
                
        except Exception as e:
            pass
            
    if best_edges is None:
        # Fallback: K_{1,19}
        best_edges = [[0, i] for i in range(1, 20)]
        
    return best_edges

# EVOLVE-BLOCK-END
