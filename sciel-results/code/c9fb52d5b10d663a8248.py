import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Helper: Check if adding edge (u, v) creates a K4
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            w = (c & -c).bit_length() - 1
            c &= c - 1
            if adj[w] & (common & ~(1 << w)):
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Helper: Find all triangles and return their bitmasks
    def get_triangles(adj):
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    w = (common & -common).bit_length() - 1
                    common &= common - 1
                    mask = (1 << u) | (1 << v) | (1 << w)
                    triangles.append(mask)
        return triangles

    # Fast evaluation: check if any subset of size k is triangle-free
    # Uses bitmask enumeration for small k
    def evaluate(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Build a set of triangle masks for fast lookup
        tri_set = set(triangles)
        
        # For each k from n down to 1, check if there's a k-subset with no triangle
        # For k close to n, use complement: a k-subset is triangle-free iff
        # its complement (size n-k) "hits" every triangle
        # For small n-k, enumerate subsets of size n-k
        
        # Strategy: enumerate subsets S of size m = n-k
        # S is a "triangle transversal" if every triangle has at least one vertex in S
        # Then the complement of S is triangle-free of size k = n-m
        
        # Find minimum m such that there exists a triangle transversal of size m
        # The answer is n - m_min
        
        # Enumerate m from 1 upward
        # For m=1: check if any single vertex hits all triangles
        # For m=2: check if any pair hits all triangles
        # etc.
        
        for m in range(1, n + 1):
            # Check if there's a subset of size m that intersects every triangle
            # Use recursive backtracking with pruning
            found = False
            
            def backtrack(start, chosen, remaining_tris):
                nonlocal found
                if found:
                    return
                if len(chosen) == m:
                    # Check if chosen hits all remaining triangles
                    for tri in remaining_tris:
                        hit = False
                        for v in chosen:
                            if tri & (1 << v):
                                hit = True
                                break
                        if not hit:
                            return
                    found = True
                    return
                
                # Pruning: if remaining_tris is empty, we've hit all
                # (but we still need to pick m total, which is fine)
                
                # Try each vertex from start to n-1
                for v in range(start, n):
                    # Pruning: if adding v doesn't help reduce remaining triangles
                    # and we've already tried similar, skip (simple pruning)
                    new_remaining = []
                    for tri in remaining_tris:
                        if not (tri & (1 << v)):
                            new_remaining.append(tri)
                    
                    # If new_remaining is same length, v doesn't hit any new triangle
                    # Still might be needed, so don't prune aggressively
                    backtrack(v + 1, chosen + [v], new_remaining)
                    if found:
                        return
            
            backtrack(0, [], triangles)
            if found:
                return n - m
        
        return 0

    # Initial Graph: Random Greedy K4-free with multiple strategies
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate multiple initial candidates and pick the best
    best_adj = None
    best_score = n + 1
    
    num_restarts = 5
    for r in range(num_restarts):
        candidate = generate_random_k4_free()
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_adj = candidate[:]
        if time.time() > deadline - 5:
            break
    
    if best_adj is None:
        best_adj = generate_random_k4_free()
        best_score = evaluate(best_adj)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local Search with simulated annealing
    t = 2.0
    cooling = 0.995
    min_t = 0.01
    
    while time.time() < deadline and t > min_t:
        # Try a random move: add or remove an edge
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random edge
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
        else:
            # Try to remove a random edge
            # Find a random existing edge
            u = random.randint(0, n - 1)
            v_mask = current_adj[u]
            if not v_mask:
                continue
            # Pick a random set bit
            v = (v_mask & -v_mask).bit_length() - 1
            # Ensure we pick randomly among set bits
            # Simpler: just pick the lowest, or use a different method
            # Let's pick uniformly from set bits
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
