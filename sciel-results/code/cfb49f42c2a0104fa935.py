import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    n = 20
    
    # Precompute all triples for fast validity checking
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    num_triples = len(triples)
    
    # Helper functions
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    def is_k33_free(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if bin(common).count('1') >= 3:
                return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free"""
        # Only need to check triples that include u or v
        # A triple (a,b,c) is affected if u or v is in {a,b,c}
        # And the new common neighbors must be >= 3
        
        # Triples containing u: (u, x, y) for x < y, x,y != u
        for x in range(n):
            if x == u:
                continue
            for y in range(x+1, n):
                if y == u:
                    continue
                if x == v or y == v:
                    # Triple is (u, v, x) or (u, v, y)
                    # Check common neighbors of u, v, and the third vertex
                    third = x if y == v else y
                    common = adj[u] & adj[v] & adj[third]
                    # After adding (u,v), u and v are connected
                    # Common neighbors of u, v, third: those adjacent to all three
                    # u and v are now adjacent to each other, but that doesn't affect common neighbors
                    # Common neighbors are vertices w adjacent to u, v, and third
                    # Before adding (u,v): common was adj[u]&adj[v]&adj[third]
                    # After adding (u,v): same, because u and v being adjacent doesn't change their other neighbors
                    # So we just check if current common >= 3
                    if bin(common).count('1') >= 3:
                        return False
                else:
                    # Triple is (u, x, y), neither is v
                    # Adding (u,v) doesn't affect this triple's common neighbors
                    # because v is not in the triple
                    pass
        
        # Triples containing v but not u: (v, x, y) for x < y, x,y != v, and u not in {x,y}
        # But we already counted triples with both u and v above
        # Triples with v but not u: these are unaffected by adding (u,v)
        # because u is not in the triple
        
        # Actually, the key insight: adding edge (u,v) only affects triples that contain BOTH u and v
        # No wait, that's not right. Adding (u,v) changes adj[u] and adj[v].
        # A triple (a,b,c) has common neighbors = adj[a] & adj[b] & adj[c]
        # If u is in the triple, adj[u] changes (v is added to adj[u])
        # So common neighbors of (u,x,y) = (adj[u] | {v}) & adj[x] & adj[y]
        # = (adj[u] & adj[x] & adj[y]) | ({v} & adj[x] & adj[y])
        # So the new common neighbors = old common neighbors + (v if v is adjacent to x and y)
        # So we need to check: old_common + (1 if v adj x and y else 0) >= 3
        
        # Let me redo this properly
        for x in range(n):
            if x == u:
                continue
            for y in range(x+1, n):
                if y == u:
                    continue
                # Triple is (u, x, y)
                old_common = adj[u] & adj[x] & adj[y]
                old_count = bin(old_common).count('1')
                # After adding (u,v): adj[u] gains v
                # New common = (adj[u] | (1<<v)) & adj[x] & adj[y]
                # = old_common | ((1<<v) & adj[x] & adj[y])
                # So new_count = old_count + (1 if v is adj to both x and y else 0)
                if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                    new_count = old_count + 1
                else:
                    new_count = old_count
                if new_count >= 3:
                    return False
        
        # Similarly for triples containing v but not u: (v, x, y) where u not in {x,y}
        # Adding (u,v) adds u to adj[v]
        # New common of (v,x,y) = old_common + (1 if u adj to x and y else 0)
        for x in range(n):
            if x == v or x == u:
                continue
            for y in range(x+1, n):
                if y == v or y == u:
                    continue
                # Triple is (v, x, y), u not in triple
                old_common = adj[v] & adj[x] & adj[y]
                old_count = bin(old_common).count('1')
                if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                    new_count = old_count + 1
                else:
                    new_count = old_count
                if new_count >= 3:
                    return False
        
        return True
    
    def can_remove_edge(adj, u, v):
        """Removing an edge always keeps the graph valid (or makes it more valid)"""
        return True
    
    # Main algorithm: Simulated Annealing
    best_edges = 0
    best_adj = None
    
    # Start with a random dense graph
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Initialize with random graph
        p = 0.5
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Make it valid by removing edges
        changed = True
        while changed:
            changed = False
            for i, j, k in triples:
                common = adj[i] & adj[j] & adj[k]
                if bin(common).count('1') >= 3:
                    # Find an edge to remove
                    # Remove a random edge from the K3,3
                    neighbors = []
                    c = common
                    while len(neighbors) < 3 and c:
                        lsb = c & -c
                        neighbors.append(lsb.bit_length() - 1)
                        c -= lsb
                    # Pick a random pair from {i,j,k} x {neighbors}
                    all_vertices = [i, j, k] + neighbors
                    random.shuffle(all_vertices)
                    for idx in range(len(all_vertices) - 1):
                        for jdx in range(idx+1, len(all_vertices)):
                            a, b = all_vertices[idx], all_vertices[jdx]
                            if (adj[a] >> b) & 1:
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
        
        # Simulated Annealing
        temperature = 10.0
        cooling_rate = 0.995
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
        
        sa_steps = 0
        while time.time() < deadline and temperature > 0.1:
            sa_steps += 1
            
            # Choose a random edge to toggle
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            edge_exists = (adj[u] >> v) & 1
            
            if edge_exists:
                # Try removing
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_edges = current_edges - 1
                # Removing always valid
                delta = new_edges - current_edges
                # Always accept removal? No, use SA
                if delta >= 0 or random.random() < pow(2.718281828, delta / temperature):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Try adding
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_edges = current_edges + 1
                    delta = new_edges - current_edges
                    if delta >= 0 or random.random() < pow(2.718281828, delta / temperature):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                # else: cannot add, skip
            
            temperature *= cooling_rate
        
        # After SA, do a greedy fill
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
        
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
        
        # Local search with kicks
        ls_start = time.time()
        while time.time() < deadline and (time.time() - ls_start) < 5:
            # Try to add any missing edge
            added = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                if added:
                    break
            
            if added:
                current_edges = count_edges(adj)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                continue
            
            # Kick: remove a random edge, then try to re-add
            existing = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
            
            if not existing:
                break
            
            u1, v1 = random.choice(existing)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            # Try to add back and others
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            
            current_edges = count_edges(adj)
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
    
    # Return best result
    if best_adj:
        return get_edges(best_adj)
    else:
        # Fallback: empty graph
        return []
