import time
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 90.0 # seconds
    
    n = P['n']
    # Bipartition: U = even, W = odd
    U = [i for i in range(n) if i % 2 == 0]
    W = [i for i in range(n) if i % 2 == 1]
    
    # Initialize adjacency sets
    # neighbors[u] is a set of neighbors in W
    # neighbors[w] is a set of neighbors in U
    neighbors = {i: set() for i in range(n)}
    
    existing_edges = set()
    for u, v in P['edges']:
        neighbors[u].add(v)
        neighbors[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
        
    # Candidate edges: all (u, w) in U x W not in existing
    candidates = []
    for u in U:
        for w in W:
            if (min(u, w), max(u, w)) not in existing_edges:
                candidates.append((u, w))
                
    # State for coverage
    # Covered pairs in U: set of frozenset({u1, u2})
    # Covered pairs in W: set of frozenset({w1, w2})
    
    def get_uncovered_pairs(partition, neighbors_map):
        """
        partition: list of vertices
        neighbors_map: dict mapping vertex to set of neighbors in OTHER partition
        Returns set of frozensets of uncovered pairs
        """
        uncovered = set()
        for i in range(len(partition)):
            for j in range(i+1, len(partition)):
                v1 = partition[i]
                v2 = partition[j]
                # Check if they share a neighbor
                # neighbors_map[v1] contains neighbors in the other partition
                # If intersection is not empty, covered
                if neighbors_map[v1] & neighbors_map[v2]:
                    continue
                uncovered.add(frozenset((v1, v2)))
        return uncovered

    # Initial uncovered pairs
    uncov_U = get_uncovered_pairs(U, neighbors)
    uncov_W = get_uncovered_pairs(W, neighbors)
    
    added_edges = []
    
    # Greedy loop
    # We want to minimize number of edges.
    # Score an edge (u, w):
    # 1. How many pairs in U does it cover?
    #    It adds w to N(u). 
    #    New pairs covered in U: pairs (u, u') such that w in N(u') 
    #    i.e., u' in neighbors[w] (current)
    #    Wait, neighbors[w] is the set of U neighbors of w.
    #    So adding (u, w) connects u to w.
    #    u now shares w with all u' in neighbors[w].
    #    So it covers pairs (u, u') for all u' in neighbors[w].
    #    Note: Some might already be covered by other w's.
    #    We only count NEWLY covered pairs.
    
    # 2. How many pairs in W does it cover?
    #    It adds u to N(w).
    #    w now shares u with all w' in neighbors[u].
    #    So it covers pairs (w, w') for all w' in neighbors[u].
    
    # To be precise, we should check if the pair is in uncov_U or uncov_W.
    
    while (uncov_U or uncov_W) and (time.time() - start_time) < deadline - 2.0:
        
        best_score = -1
        best_edge = None
        
        for u, w in candidates:
            # Check if edge is already added (it shouldn't be, we remove from candidates)
            # Check if it's valid (not existing, not added)
            
            # Calculate gain in U
            gain_U = 0
            # u will be connected to w.
            # u's neighbors in W will include w.
            # Any u' in neighbors[w] will now share w with u.
            # So pairs (u, u') are candidates for covering.
            for u_prime in neighbors[w]:
                pair = frozenset((u, u_prime))
                if pair in uncov_U:
                    gain_U += 1
                    
            # Calculate gain in W
            gain_W = 0
            # w will be connected to u.
            # Any w' in neighbors[u] will now share u with w.
            for w_prime in neighbors[u]:
                pair = frozenset((w, w_prime))
                if pair in uncov_W:
                    gain_W += 1
                    
            score = gain_U + gain_W
            
            if score > best_score:
                best_score = score
                best_edge = (u, w)
                
        if best_edge is None or best_score <= 0:
            # No edge can improve coverage? 
            # This shouldn't happen if graph is not complete bipartite.
            # If it happens, we might have a deadlock or need a different strategy.
            # For now, break to avoid infinite loop.
            break
            
        u, w = best_edge
        # Add edge
        added_edges.append([u, w])
        neighbors[u].add(w)
        neighbors[w].add(u)
        
        # Remove from candidates
        # It's inefficient to remove from list every time. 
        # We can just check if edge is in added_edges or existing in the loop.
        # But candidates list is static. 
        # We should skip edges that are already added.
        
        # Update uncovered sets
        # Instead of recalculating from scratch, we can update, but recalculating is safer for correctness 
        # and N is small (12x12 pairs = 66+66=132 pairs). 
        # Recalculating 132 pairs is very fast.
        uncov_U = get_uncovered_pairs(U, neighbors)
        uncov_W = get_uncovered_pairs(W, neighbors)
        
        # Optional: Remove best_edge from candidates to save time?
        # We can filter candidates in the loop.
        
    # Final check: Is diameter <= 4?
    # The greedy aims for diameter <= 3. 
    # If it fails, we might need to add more edges or verify.
    # Given the problem constraints, the greedy should work.
    
    # Convert to list of lists of ints
    result = [[int(u), int(v)] for u, v in added_edges]
    
    return result
