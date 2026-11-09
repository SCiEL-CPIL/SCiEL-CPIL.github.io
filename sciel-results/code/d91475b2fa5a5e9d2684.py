"""Find a dense triangle-free graph on 16 vertices with chromatic number >= 4."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 16
    
    # Precompute all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # For triangle-freeness: if (u,v) and (v,w) are edges, (u,w) must not be
    # Precompute for each edge (u,v), the set of edges that conflict with it
    # i.e., edges (a,b) such that {a,b} shares exactly one vertex with {u,v} 
    # and the other endpoint creates a triangle
    # Actually, if we add edge (u,v), we cannot have any edge (u,w) and (v,w) for same w
    # More precisely: edge (u,v) conflicts with edge (w,x) if 
    #   (w==u and x==v) or (w==v and x==u) [same edge]
    #   or (w==u and x!=v and there's an edge (v,x)) -- this is dynamic
    #   
    # Better approach: maintain adjacency and check triangle-freeness incrementally.
    # When adding edge (u,v), it creates a triangle iff there exists w such that
    # (u,w) is already an edge AND (v,w) is already an edge.
    
    # Precompute for each pair (u,v), the list of common neighbors candidates
    # For edge (u,v): conflicting edges are those (u,w) or (v,w) where the other is also present
    # We'll use a bitset representation for speed
    
    # Represent graph as list of bitmasks: adj[u] is a bitmask of neighbors of u
    # For triangle check when adding (u,v): (adj[u] & adj[v]) != 0 means triangle exists
    
    # Strategy: local search / simulated annealing
    # Start from a known 4-chromatic triangle-free graph (Grötzsch on 11 vertices + 5 isolated)
    # Then try to add edges while maintaining triangle-free and 4-chromatic properties
    
    def groetzsch_edges():
        """Return edges of Grötzsch graph (Mycielski of C5) on vertices 0..10."""
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        edges += [[5 + u, 10] for u in range(5)]
        return edges
    
    def is_triangle_free(adj):
        """Check if graph with adjacency bitmasks is triangle-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking."""
        color = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine available colors
            used = 0
            for u in range(n):
                if adj[v] & (1 << u):
                    if color[u] != -1:
                        used |= (1 << color[u])
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def is_4_chromatic(adj):
        """Check if graph has chromatic number at least 4 (i.e., not 3-colorable)."""
        return not is_3_colorable(adj)
    
    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2
    
    def add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if successful (no triangle created)."""
        if adj[u] & (1 << v):
            return True  # already present
        # Check triangle: is there w with (u,w) and (v,w) both present?
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # Start from Grötzsch graph
    base_edges = groetzsch_edges()
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    best_adj = adj[:]
    best_count = edge_count(adj)
    
    # Verify starting graph is valid
    assert is_triangle_free(adj), "Grötzsch graph should be triangle-free"
    assert is_4_chromatic(adj), "Grötzsch graph should be 4-chromatic"
    
    # Try to add all possible edges greedily
    random.seed(42)
    
    # Phase 1: Greedy addition of edges
    for u, v in all_edges:
        if not (adj[u] & (1 << v)):
            if not (adj[u] & adj[v]):  # no triangle
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Now try to remove edges to make it 3-colorable, then re-add
    # Actually, we want to maximize edges while keeping 4-chromatic and triangle-free
    
    # Phase 2: Local search with simulated annealing
    # Move types: add an edge, remove an edge, swap (remove one, add another)
    
    current_adj = adj[:]
    current_count = edge_count(current_adj)
    
    # Check if current is 4-chromatic
    current_4chrom = is_4_chromatic(current_adj)
    
    if not current_4chrom:
        # Need to make it 4-chromatic. Start fresh from Grötzsch.
        current_adj = [0] * n
        for u, v in base_edges:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
        current_count = edge_count(current_adj)
        current_4chrom = True
        best_adj = current_adj[:]
        best_count = current_count
    
    # Simulated annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # Precompute edge list for random selection
    def random_move(adj):
        """Generate a random move: add, remove, or swap an edge."""
        r = random.random()
        if r < 0.3:
            # Add an edge
            u, v = random.choice(all_edges)
            if not (adj[u] & (1 << v)):
                if adj[u] & adj[v]:
                    return None  # would create triangle
                return ('add', u, v)
        elif r < 0.6:
            # Remove an edge
            u, v = random.choice(all_edges)
            if adj[u] & (1 << v):
                return ('remove', u, v)
        else:
            # Swap: remove one, add another
            u1, v1 = random.choice(all_edges)
            u2, v2 = random.choice(all_edges)
            if adj[u1] & (1 << v1) and not (adj[u2] & (1 << v2)):
                if not (adj[u2] & adj[v2]):
                    return ('swap', u1, v1, u2, v2)
        return None
    
    iterations = 0
    while time.time() - start_time < DEADLINE:
        move = random_move(current_adj)
        if move is None:
            iterations += 1
            continue
        
        old_adj = current_adj[:]
        old_count = current_count
        old_4chrom = current_4chrom
        
        if move[0] == 'add':
            _, u, v = move
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            delta_count = 1
        elif move[0] == 'remove':
            _, u, v = move
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            delta_count = -1
        else:  # swap
            _, u1, v1, u2, v2 = move
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            delta_count = 0
        
        new_count = old_count + delta_count
        
        # Check triangle-free
        if not is_triangle_free(current_adj):
            current_adj = old_adj
            iterations += 1
            continue
        
        # Check 4-chromatic (expensive)
        # Only check if count changed or periodically
        new_4chrom = is_4_chromatic(current_adj)
        
        # Acceptance criterion
        if new_4chrom:
            if new_count >= old_count:
                # Accept
                pass
            else:
                # Accept with probability exp(delta/T)
                delta = new_count - old_count
                if random.random() < pow(2.718281828, delta / T):
                    pass
                else:
                    current_adj = old_adj
                    iterations += 1
                    continue
        else:
            # Not 4-chromatic: reject unless we're in a very hot phase
            # Actually, we need to maintain 4-chromaticity
            current_adj = old_adj
            iterations += 1
            continue
        
        current_count = new_count
        current_4chrom = new_4chrom
        
        if current_count > best_count:
            best_adj = current_adj[:]
            best_count = current_count
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        iterations += 1
        if iterations % 10000 == 0:
            # Periodic restart if stuck
            if T <= min_T * 1.1:
                T = 0.5
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
