import time
import random
import sys

def construct():
    """
    Constructs a K2,3-free graph on 20 vertices maximizing edges.
    Uses Simulated Annealing with bitmask adjacency for speed.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 20
    
    # Initialize adjacency matrix as list of integers (bitmasks)
    # adj[i] has bit j set if edge (i, j) exists
    adj = [0] * n
    edge_count = 0
    best_adj = adj[:]
    best_count = 0

    # Helper to add edge
    def add_edge(u, v):
        nonlocal edge_count
        if not (adj[u] & (1 << v)):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_count += 1
            return True
        return False

    # Helper to remove edge
    def remove_edge(u, v):
        nonlocal edge_count
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edge_count -= 1
            return True
        return False

    # Helper to check if adding edge (u, v) violates K2,3
    # Violation if common neighbors of u and v >= 3
    # Common neighbors = adj[u] & adj[v]
    # We need to check if bin(common).count('1') >= 3
    def can_add(u, v):
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        # Popcount
        # Python 3.10+ has int.bit_count(), but for compatibility we can use bin(x).count('1')
        # Given 120s limit, bin().count() is fast enough for 20-bit ints.
        if common.bit_count() >= 3:
            return False
        return True

    # Helper to check if removing edge (u, v) is valid (always valid for K2,3)
    # But we only remove if it helps. Actually, removing an edge never creates a K2,3.
    # So any edge can be removed.

    # Greedy initialization
    # Try to add edges in a random order, prioritizing those that don't block many others?
    # Simple greedy: iterate all pairs, add if possible.
    # To diversify, we can shuffle the order of pairs.
    
    # Let's run multiple restarts or a single long SA.
    # Given 110s, a single long SA with good restarts is better.
    
    # Initialize with a random greedy construction
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Reset state
    adj = [0] * n
    edge_count = 0
    
    # Greedy fill
    random.shuffle(pairs)
    for u, v in pairs:
        if can_add(u, v):
            add_edge(u, v)
    
    best_adj = adj[:]
    best_count = edge_count

    # Simulated Annealing
    # State: adj, edge_count
    # Move: Add a random non-existent edge if possible, or Remove a random existing edge, or Swap?
    # Swap is complex. Let's stick to Add/Remove.
    # To escape local optima, we might need to remove edges even if it lowers count, 
    # but since we maximize edges, removing is bad unless it allows adding more later.
    # Standard SA: accept worse moves with probability exp(-delta/T).
    # Here "worse" means lower edge count.
    # Delta = new_count - old_count.
    
    T = 10.0
    cooling = 0.99999 # Very slow cooling
    min_T = 0.01
    
    # Precompute list of all possible edges for random selection
    all_edges = pairs
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a move
        # 1. Pick a random edge
        u, v = random.choice(all_edges)
        
        if adj[u] & (1 << v):
            # Edge exists, remove it
            remove_edge(u, v)
            new_count = edge_count
            delta = new_count - best_count # Not quite, delta should be vs current state
            # Wait, SA works on current state.
            # Let's track current_count separately from best.
            # Actually, the loop above modified adj and edge_count.
            # Let's restructure.
            
            # This logic is getting messy with nonlocal in nested loops.
            # Let's rewrite the SA loop cleanly.
            pass
        else:
            # Edge doesn't exist, try to add
            if can_add(u, v):
                add_edge(u, v)
            else:
                # Cannot add. Maybe remove a different edge to allow this one?
                # That's a 2-move. Too complex for now.
                pass
        T *= cooling

    # The above SA loop was flawed in implementation sketch. 
    # Let's implement a proper SA loop.
    
    # Reset to best found so far (which is the greedy one)
    adj = best_adj[:]
    edge_count = best_count
    
    T = 5.0
    cooling = 0.9999
    min_T = 0.01
    
    # To make SA effective, we need to be able to accept "worse" states (fewer edges).
    # So we track current_count.
    current_count = edge_count
    
    while time.time() < deadline:
        if T < min_T:
            # Reheat or break? Break is safer.
            break
            
        u, v = random.choice(all_edges)
        
        if adj[u] & (1 << v):
            # Remove edge
            remove_edge(u, v)
            new_count = edge_count
            delta = new_count - current_count
            
            # Acceptance probability
            if delta >= 0:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Worse move
                # Probability exp(delta / T)
                if random.random() < (2.71828 ** (delta / T)):
                    current_count = new_count
                else:
                    # Reject: restore edge
                    add_edge(u, v)
        else:
            # Add edge
            if can_add(u, v):
                add_edge(u, v)
                new_count = edge_count
                delta = new_count - current_count
                
                if delta >= 0:
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # This case shouldn't happen if can_add is correct and we only add valid edges
                    # Adding a valid edge always increases count by 1.
                    current_count = new_count
            else:
                # Cannot add. 
                # Strategy: Try to remove a random edge that is a "bottleneck" for this pair?
                # Or just skip.
                # To improve diversity, we could try to remove one of the common neighbors' edges?
                # For now, skip.
                pass
        
        T *= cooling

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        v = 0
        while mask:
            if mask & 1:
                if u < v:
                    edges.append([u, v])
            mask >>= 1
            v += 1
            
    return edges
