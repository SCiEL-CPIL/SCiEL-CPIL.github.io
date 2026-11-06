import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        """Return list of triangle bitmasks."""
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def evaluate(adj):
        """
        Find the size of the largest triangle-free induced subgraph.
        A k-subset is triangle-free iff its complement (size m=n-k) is a
        triangle transversal (hits every triangle).
        Find minimum m such that a transversal of size m exists; answer = n - m.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Remove duplicate triangles (shouldn't happen but safe)
        triangles = list(set(triangles))
        
        # For m from 1 upward, check if there's a subset of size m hitting all triangles
        for m in range(1, n + 1):
            found = False
            
            def backtrack(start, chosen_count, hit_mask):
                """
                hit_mask: bitmask of triangles already hit (indexed by position)
                We use a different approach: track which triangles remain unhit.
                """
                nonlocal found
                if found:
                    return
                if chosen_count == m:
                    # All triangles must be hit - check if any remain
                    # We track remaining via the recursive structure
                    # Actually, let's use a cleaner approach below
                    return
                
                # Pruning: if we can't possibly hit all remaining triangles
                # with the remaining slots, return early
                # (simple version: just proceed)
                
                for v in range(start, n):
                    if chosen_count == m:
                        break
                    # Try adding vertex v
                    new_chosen_count = chosen_count + 1
                    backtrack(v + 1, new_chosen_count, hit_mask)
                    if found:
                        return
            
            # Better approach: use bitmask for triangles
            # Represent each triangle by its index, track which are hit
            num_tris = len(triangles)
            if num_tris == 0:
                return n
            
            # For each vertex, precompute which triangles it hits
            vertex_hits = [0] * n
            for v in range(n):
                mask = 1 << v
                hit = 0
                for i, tri in enumerate(triangles):
                    if tri & mask:
                        hit |= (1 << i)
                vertex_hits[v] = hit
            
            all_tris_mask = (1 << num_tris) - 1
            
            def backtrack2(start, chosen, covered):
                nonlocal found
                if found:
                    return
                if covered == all_tris_mask:
                    found = True
                    return
                if len(chosen) == m:
                    return
                
                # Pruning: even if we pick all remaining vertices, can we cover all?
                # Simple: just try
                for v in range(start, n):
                    new_covered = covered | vertex_hits[v]
                    if new_covered == covered:
                        # v doesn't help, skip (optimization)
                        continue
                    backtrack2(v + 1, chosen + [v], new_covered)
                    if found:
                        return
            
            backtrack2(0, [], 0)
            if found:
                return n - m
        
        return 1

    # Initialize with K6,6,6 complete tripartite
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    # Also try a few random K4-free initializations
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate multiple initial candidates
    best_adj = None
    best_score = n + 1
    
    # Start with K6,6,6
    candidate = init_k666()
    score = evaluate(candidate)
    best_score = score
    best_adj = candidate[:]
    
    # Try a few random starts
    for r in range(3):
        if time.time() > deadline - 5:
            break
        candidate = generate_random_k4_free()
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_adj = candidate[:]
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local search: focus on adding edges (reduces triangle-free subgraphs)
    # and occasional removals to escape local minima
    t = 1.5
    cooling = 0.998
    min_t = 0.01
    
    while time.time() < deadline and t > min_t:
        # Try to add a random edge (most likely to improve)
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if current_adj[u] & (1 << v):
            continue
        if not can_add_edge(current_adj, u, v):
            continue
        
        old_u = current_adj[u]
        old_v = current_adj[v]
        add_edge(current_adj, u, v)
        new_score = evaluate(current_adj)
        
        delta = new_score - current_score
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            current_adj[u] = old_u
            current_adj[v] = old_v
        
        # Occasionally try removing an edge to explore
        if random.random() < 0.15:
            u = random.randint(0, n - 1)
            v_mask = current_adj[u]
            if v_mask:
                # Pick a random neighbor
                bits = []
                tmp = v_mask
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    bits.append(b)
                    tmp &= tmp - 1
                v = random.choice(bits)
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                delta = new_score - current_score
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
        
        t *= cooling
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
