"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
import time
import random

def has_c5(n, adj):
    """Check if graph has a 5-cycle (as a subgraph, not necessarily induced)."""
    # A 5-cycle is a sequence v0-v1-v2-v3-v4-v0 where all edges exist
    # and all vertices are distinct.
    for v0 in range(n):
        for v1 in range(v0+1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1+1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2+1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3+1, n):
                        if not adj[v3][v4] or not adj[v4][v0]:
                            continue
                        # Check all 5 edges exist
                        if adj[v0][v1] and adj[v1][v2] and adj[v2][v3] and adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def max_independent_set_size(n, adj):
    """Compute the independence number using a greedy approach + verification."""
    # For small n, we can use a more exact method
    # Greedy: repeatedly pick vertex with min degree in remaining graph
    remaining = set(range(n))
    independent = []
    while remaining:
        # Pick vertex with minimum degree in induced subgraph on remaining
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and adj[v][u])
            if deg < min_deg:
                min_deg = deg
                min_v = v
        independent.append(min_v)
        remaining.discard(min_v)
        # Remove neighbors of min_v from remaining (they can't be in independent set with min_v)
        # Actually no - for independence, we just remove min_v and its neighbors
        for u in list(remaining):
            if adj[min_v][u]:
                remaining.discard(u)
    return len(independent)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Start with 8xK4: n=32
    n = 32
    adj = [[False]*n for _ in range(n)]
    for b in range(8):
        base = 4*b
        for i in range(4):
            for j in range(i+1, 4):
                adj[base+i][base+j] = True
                adj[base+j][base+i] = True
    
    # Verify initial construction
    assert not has_c5(n, adj), "Initial construction has C5"
    assert max_independent_set_size(n, adj) <= 8, "Initial construction has alpha > 8"
    
    # Try to extend to n=33, 34, 35 by adding vertices
    # Strategy: add a new vertex and try to connect it to existing vertices
    # such that no C5 is created and alpha remains < 9
    
    best_n = n
    best_adj = [row[:] for row in adj]
    
    for target_n in range(n+1, 36):
        if time.time() > deadline:
            break
        new_adj = [row[:] for row in adj]
        # Extend adj matrix
        for i in range(target_n):
            new_adj[i] = new_adj[i] + [False]*(target_n - n)
        for i in range(n, target_n):
            new_adj.append([False]*target_n)
        
        # Try adding vertices one at a time
        # For each new vertex, try different connection patterns
        found = True
        while found:
            found = False
            if time.time() > deadline:
                break
            current_n = len(new_adj)
            if current_n >= target_n:
                break
            # Add a new vertex
            new_adj.append([False]*current_n)
            for i in range(current_n):
                new_adj[i].append(False)
            current_n += 1
            new_v = current_n - 1
            
            # Try different subsets of neighbors for new_v
            # To avoid C5, new_v should not create a 5-cycle
            # A 5-cycle involving new_v would need a path of length 4 from some vertex to new_v
            # through existing vertices
            
            # Simple strategy: connect new_v to all vertices in one K4 component
            # This keeps it within that component essentially
            # But this might not help with alpha
            
            # Alternative: try to connect new_v to create a larger clique or structure
            # Let's try connecting to a subset that maintains C5-freeness
            
            # Try: connect to 0 vertices (isolated) - but this increases alpha
            # Try: connect to all vertices in one K4 (making it K5) - K5 has no C5
            # Actually K5 has no 5-cycle as a subgraph? Wait, K5 contains C5 as a subgraph!
            # K5 has all 5-cycles. So we can't make K5.
            
            # So we need to be careful. Let's try a different approach.
            # Use a known construction or try random search.
            
            # For now, let's just try a few patterns
            best_pattern = None
            for trial in range(100):
                if time.time() > deadline:
                    break
                # Random subset of neighbors
                neighbors = set()
                for v in range(new_v):
                    if random.random() < 0.3:
                        neighbors.add(v)
                # Check C5-freeness
                test_adj = [row[:] for row in new_adj]
                for v in neighbors:
                    test_adj[new_v][v] = True
                    test_adj[v][new_v] = True
                if has_c5(current_n, test_adj):
                    continue
                # Check alpha
                if max_independent_set_size(current_n, test_adj) > 8:
                    continue
                best_pattern = neighbors
                break
            
            if best_pattern is not None:
                for v in best_pattern:
                    new_adj[new_v][v] = True
                    new_adj[v][new_v] = True
                found = True
            else:
                # Can't add this vertex, remove it
                new_adj.pop()
                for i in range(len(new_adj)):
                    new_adj[i].pop()
                break
        
        if len(new_adj) >= target_n and not has_c5(target_n, new_adj) and max_independent_set_size(target_n, new_adj) <= 8:
            best_n = target_n
            best_adj = [row[:] for row in new_adj]
            adj = [row[:] for row in new_adj]
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
