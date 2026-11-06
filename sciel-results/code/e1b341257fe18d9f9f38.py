import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    n = 31
    q = 5
    target_edges = 91
    
    # Generate points of PG(2, q) using homogeneous coordinates
    # Points: [x:y:z] where not all zero, modulo scalar multiplication
    # Use canonical form: first nonzero coordinate is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first nonzero component 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    pt = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    pt = (0, 1, (z * inv_y) % q)
                else:
                    pt = (0, 0, 1)
                
                if pt not in points:
                    points.append(pt)
    
    if len(points) != n:
        # Fallback: use a different enumeration
        points = []
        # Points at infinity: [x:1:0] for x in 0..q-1, and [1:0:0]
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        # Finite points: [x:y:1] for x, y in 0..q-1
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        
        if len(points) != n:
            # Final fallback: random graph
            import random
            random.seed(42)
            edges = []
            edge_set = set()
            while len(edges) < target_edges:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    edges.append([u, v])
            return edges
    
    # Build adjacency using polarity: dot(p, q) = p[0]*q[0] + p[1]*q[1] + p[2]*q[2] ≡ 0 (mod 5)
    neighbors = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]) % q
            if dot == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.append([i, j])
    
    current_edges = len(edges)
    
    # Find all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in neighbors[i]:
                non_edges.append((i, j))
    
    edges_to_add = target_edges - current_edges
    
    if edges_to_add <= 0:
        # If we already have enough edges, trim to exactly target
        return edges[:target_edges]
    
    # For each non-edge, compute the number of 4-cycles that would be created
    # A 4-cycle through new edge (u,v): u-a-b-v-u where a~u, b~v, a~b, all distinct
    # Count = number of pairs (a,b) with a in N(u), b in N(v), a != b, a != v, b != u, and a~b
    
    def count_new_c4s(u, v):
        """Count 4-cycles created by adding edge (u,v)."""
        count = 0
        for a in neighbors[u]:
            if a == v:
                continue
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if b in neighbors[a]:
                    count += 1
        return count
    
    # Find the non-edge with minimum new C4s
    best_edge = None
    min_cost = float('inf')
    
    for u, v in non_edges:
        if time.time() - start_time > DEADLINE - 10:
            break
        cost = count_new_c4s(u, v)
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
    
    if best_edge is None:
        # Fallback: add first non-edge
        if non_edges:
            best_edge = non_edges[0]
        else:
            return edges[:target_edges]
    
    # Add the best edge
    u, v = best_edge
    edges.append([u, v])
    neighbors[u].add(v)
    neighbors[v].add(u)
    
    # Now we have exactly 91 edges. Use simulated annealing to try to reduce C4 count.
    # The total number of C4s can be computed as: for each pair of vertices, 
    # C4s = (1/2) * sum over all unordered pairs {u,v} of (common_neighbors(u,v) choose 2)
    # Wait, that's not quite right. The number of 4-cycles is:
    # For each 4-set of vertices {a,b,c,d}, count if it forms a C4.
    # More efficiently: sum over all unordered pairs {u,v} of C(common(u,v), 2) / 2
    # because each C4 has two pairs of opposite vertices.
    
    def count_total_c4s():
        """Count total number of 4-cycles in the graph."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = len(neighbors[i] & neighbors[j])
                total += common * (common - 1) // 2
        return total // 2
    
    initial_c4 = count_total_c4s()
    
    # Simulated annealing: try to swap edges to reduce C4 count
    # Move: remove edge (a,b), add edge (c,d) where (c,d) is a non-edge
    # Accept if it reduces C4 count, or with probability exp(-delta/T)
    
    import random
    random.seed(42)
    
    best_c4 = initial_c4
    best_edges = [e[:] for e in edges]
    best_neighbors = [set(nb) for nb in neighbors]
    
    T = 50.0  # Initial temperature
    cooling = 0.995
    T_min = 0.5
    
    current_c4 = initial_c4
    cur_neighbors = [set(nb) for nb in neighbors]
    cur_edges = [e[:] for e in edges]
    
    # Precompute all possible non-edge pairs for quick access
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iter_count = 0
    while time.time() - start_time < DEADLINE and T > T_min:
        iter_count += 1
        
        # Pick a random edge to remove
        edge_idx = random.randint(0, len(cur_edges) - 1)
        a, b = cur_edges[edge_idx]
        
        # Pick a random non-edge to add
        # Find non-edges: pairs not in cur_neighbors
        # For efficiency, pick random pair and check
        found_non_edge = False
        attempts = 0
        while attempts < 100:
            c, d = random.sample(range(n), 2)
            if c > d:
                c, d = d, c
            if d not in cur_neighbors[c]:
                found_non_edge = True
                break
            attempts += 1
        
        if not found_non_edge:
            T *= cooling
            continue
        
        # Check that we're not just swapping the same edge back
        if (c == a and d == b) or (c == b and d == a):
            T *= cooling
            continue
        
        # Compute change in C4 count
        # This is complex; approximate by computing delta
        # Remove (a,b): affects C4s involving a and b
        # Add (c,d): affects C4s involving c and d
        
        # For efficiency, compute the delta directly
        # A C4 is counted if it uses the edge being removed or added
        
        # Delta from removing (a,b):
        # C4s that use edge (a,b) are those where a and b are adjacent in the C4
        # Number of C4s using edge (a,b) = number of common neighbors of a and b that are 
        # connected through a path of length 2... actually, C4s containing edge (a,b) 
        # are: for each common neighbor x of a and b, and each other vertex y such that 
        # y~a and y~b... no, that's not right either.
        
        # Simpler: just compute the full C4 count before and after for small changes
        # But that's O(n^2) per iteration which might be too slow.
        
        # Alternative: use the formula C4 = (1/2) * sum_{u<v} C(common(u,v), 2)
        # When we remove edge (a,b), common(u,v) changes only for pairs involving a or b
        # Specifically, for any vertex w, common(a,w) decreases by 1 if b~w, 
        # and common(b,w) decreases by 1 if a~w.
        
        # Let's compute delta more carefully
        # Before: C4_count = (1/2) * sum_{u<v} C(cn(u,v), 2) where cn(u,v) = |N(u) ∩ N(v)|
        
        # After removing (a,b):
        # For pair (a, w) where w != b: cn(a,w) decreases by 1 if b in N(w), i.e., if w in N(b)
        # For pair (b, w) where w != a: cn(b,w) decreases by 1 if a in N(w), i.e., if w in N(a)
        # For pair (a, b): cn(a,b) decreases by 0 (they're no longer adjacent, but cn is about common neighbors)
        # Wait, cn(a,b) = |N(a) ∩ N(b)|. Removing edge (a,b) doesn't change N(a) ∩ N(b) 
        # because N(a) and N(b) don't include each other in the common neighbor count... 
        # Actually N(a) includes b if a~b, but when we remove the edge, b is removed from N(a) 
        # and a is removed from N(b). So cn(a,b) = |N(a) ∩ N(b)| changes.
        
        # This is getting complex. Let me just compute the C4 count directly for now,
        # but limit the number of SA iterations.
        
        # Actually, let me just do a few greedy improvements instead of full SA
        # Try each possible swap and accept if it improves
        
        if iter_count % 100 == 0:
            # Periodically try greedy swaps
            improved = True
            while improved and time.time() - start_time < DEADLINE:
                improved = False
                for i in range(len(cur_edges)):
                    if time.time() - start_time > DEADLINE:
                        break
                    a, b = cur_edges[i]
                    # Try to find a non-edge to add that reduces C4
                    # For now, just try a few random non-edges
                    for attempt in range(5):
                        c, d = random.sample(range(n), 2)
                        if c > d:
                            c, d = d, c
                        if d in cur_neighbors[c]:
                            continue
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        
                        # Tentatively make the swap
                        cur_neighbors[a].discard(b)
                        cur_neighbors[b].discard(a)
                        cur_neighbors[c].add(d)
                        cur_neighbors[d].add(c)
                        
                        new_c4 = count_total_c4s()
                        
                        if new_c4 < current_c4:
                            current_c4 = new_c4
                            cur_edges[i] = [c, d]
                            if new_c4 < best_c4:
                                best_c4 = new_c4
                                best_edges = [e[:] for e in cur_edges]
                                best_neighbors = [set(nb) for nb in cur_neighbors]
                            improved = True
                        else:
                            # Revert
                            cur_neighbors[a].add(b)
                            cur_neighbors[b].add(a)
                            cur_neighbors[c].discard(d)
                            cur_neighbors[d].discard(c)
                        
                        if improved:
                            break
        
        T *= cooling
    
    # Return the best solution found
    if best_c4 < initial_c4:
        return best_edges
    else:
        return edges
