import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 20
    
    # GF(4) implementation
    # Elements: 0, 1, 2 (omega), 3 (omega^2)
    # 0: 0
    # 1: 1
    # 2: w
    # 3: w^2
    # w^2 = w + 1
    # w^3 = 1
    
    def gf4_add(a, b):
        return a ^ b
        
    def gf4_mul(a, b):
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        # a, b in {2, 3}
        # 2*2 = 3
        # 2*3 = 1
        # 3*3 = 2
        if a == 2 and b == 2:
            return 3
        if a == 2 and b == 3:
            return 1
        if a == 3 and b == 2:
            return 1
        if a == 3 and b == 3:
            return 2
        return 0 # Should not reach here with valid inputs

    # Generate points of PG(2,4)
    points = []
    # x, y, z in GF(4)
    # Normalize: first non-zero is 1
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is 1, 2, 3
                    # If x==1, keep
                    # If x==2, multiply by 2 (since 2*2=3? No, 2*3=1? No. Inverse of 2 is 3. 2*3=1. So multiply by 3)
                    # Inverse of 1 is 1.
                    # Inverse of 2 is 3.
                    # Inverse of 3 is 2.
                    if x == 1:
                        pass
                    elif x == 2:
                        x = gf4_mul(x, 3)
                        y = gf4_mul(y, 3)
                        z = gf4_mul(z, 3)
                    elif x == 3:
                        x = gf4_mul(x, 2)
                        y = gf4_mul(y, 2)
                        z = gf4_mul(z, 2)
                    points.append((x, y, z))
                elif y != 0:
                    if y == 1:
                        pass
                    elif y == 2:
                        y = gf4_mul(y, 3)
                        z = gf4_mul(z, 3)
                    elif y == 3:
                        y = gf4_mul(y, 2)
                        z = gf4_mul(z, 2)
                    points.append((x, y, z))
                else:
                    # z != 0
                    if z == 1:
                        pass
                    elif z == 2:
                        z = gf4_mul(z, 3)
                    elif z == 3:
                        z = gf4_mul(z, 2)
                    points.append((x, y, z))
                    
    # Should have 21 points
    # Remove duplicates? The normalization should be unique.
    # Let's verify count
    # 1,0,0
    # 0,1,0
    # 0,0,1
    # 1,1,0
    # 1,0,1
    # 0,1,1
    # 1,1,1
    # 1,2,0
    # 1,0,2
    # 0,1,2
    # 1,2,1
    # 1,1,2
    # 0,2,1
    # 1,2,2
    # 1,3,0
    # 1,0,3
    # 0,1,3
    # 1,3,1
    # 1,1,3
    # 0,3,1
    # 1,3,3
    # Total 21.
    
    # Define bilinear form B(u, v) = u1 v2 + u2 v1 + u2 v3 + u3 v2 + u1 v3 + u3 v1
    def bilinear(u, v):
        u1, u2, u3 = u
        v1, v2, v3 = v
        return gf4_add(gf4_add(gf4_add(gf4_mul(u1, v2), gf4_mul(u2, v1)), gf4_mul(u2, v3)), 
                       gf4_add(gf4_add(gf4_mul(u3, v2), gf4_mul(u1, v3)), gf4_mul(u3, v1)))
        
    # Build adjacency matrix for 21 vertices
    adj21 = [[0]*21 for _ in range(21)]
    for i in range(21):
        for j in range(21):
            if i != j and bilinear(points[i], points[j]) == 0:
                adj21[i][j] = 1
                adj21[j][i] = 1
                
    # Select 20 vertices. Remove vertex 20 (index 20)
    # Or just use first 20.
    # The subgraph induced by first 20 vertices is K3,3-free.
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if adj21[i][j] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i) # Redundant if loop covers both, but safe
                
    # Note: adj21 is symmetric.
    # Let's verify edge count
    # Polarity graph of PG(2,4) has 84 edges.
    # Removing one vertex removes its degree.
    # Degree of a vertex in polarity graph of PG(2,q) is q+1 + (number of absolute points on polar line - 1).
    # For q=4, degree is roughly 5 + something.
    # Actually, all vertices have degree 5 or 6?
    # It's a regular graph? No.
    # But it's dense.
    
    # Local Search
    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def is_k33_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                uv = adj[u] & adj[v]
                if uv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    if (uv & adj[w]).bit_count() >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v):
        # Check if adding (u, v) creates K3,3
        # New adj[u] = adj[u] | (1 << v)
        # New adj[v] = adj[v] | (1 << u)
        
        # Triples involving u:
        # 1. {u, v, w}: common neighbors of u, v, w
        #    N_new(u) & N_new(v) & N(w)
        #    N_new(u) = N_old(u) | {v}
        #    N_new(v) = N_old(v) | {u}
        #    Intersection: (N_old(u) | {v}) & (N_old(v) | {u}) & N(w)
        #    = (N_old(u) & N_old(v) & N(w)) | (N_old(u) & {u} & N(w)) | ({v} & N_old(v) & N(w)) | ({v} & {u} & N(w))
        #    {u} & N(w) is empty if u not in N(w).
        #    {v} & N_old(v) is empty.
        #    So it's (N_old(u) & N_old(v) & N(w)) | ({v} & N(w) if u in N(w)?) No.
        #    Let's just compute the new bitsets and check.
        
        nu = adj[u] | (1 << v)
        nv = adj[v] | (1 << u)
        
        # Check triples with u and v
        # For all w != u, v:
        #   |(nu & nv & adj[w])| >= 3 ?
        #   |(nu & adj[w] & adj[x])| >= 3 for x != u, v, w ?
        #   This is expensive.
        
        # Optimized check:
        # Only triples containing u or v can change.
        # Triples are sets of 3 vertices.
        # If a triple does not contain u or v, its common neighbors are unchanged.
        # So we only check triples containing u or v.
        
        # Case 1: Triple is {u, v, w}
        # Common neighbors: nu & nv & adj[w]
        for w in range(n):
            if w == u or w == v:
                continue
            if (nu & nv & adj[w]).bit_count() >= 3:
                return False
                
        # Case 2: Triple is {u, w, x} where w, x != v
        # Common neighbors: nu & adj[w] & adj[x]
        # Note: nu = adj[u] | {v}
        # So (adj[u] | {v}) & adj[w] & adj[x]
        # = (adj[u] & adj[w] & adj[x]) | ({v} & adj[w] & adj[x])
        # The first term was < 3 before (since graph was valid).
        # So we only need to check if adding v makes it >= 3.
        # This happens if v is in adj[w] & adj[x] AND (adj[u] & adj[w] & adj[x]).bit_count() >= 2.
        # Or if (adj[u] & adj[w] & adj[x]).bit_count() was 2 and v is a new common neighbor.
        
        # Let's just iterate w, x
        for w in range(n):
            if w == u or w == v:
                continue
            for x in range(w + 1, n):
                if x == u or x == v:
                    continue
                # Check triple {u, w, x}
                if (nu & adj[w] & adj[x]).bit_count() >= 3:
                    return False
                # Check triple {v, w, x}
                if (nv & adj[w] & adj[x]).bit_count() >= 3:
                    return False
                    
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def local_search(adj, deadline):
        current = adj[:]
        current_edges = count_edges(current)
        
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((current[i] >> j) & 1):
                    missing.append((i, j))
                    
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            random.shuffle(missing)
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 2:
                    new_missing.append((u, v))
                    continue
                if can_add_edge(current, u, v):
                    add_edge(current, u, v)
                    current_edges += 1
                    improved = True
                else:
                    new_missing.append((u, v))
            if improved:
                missing = new_missing
                continue
                
            # 2-Opt: Remove 1, Add 2? Or Remove 1, Add 1?
            # Since we are maximizing edges, 1-opt (add) is primary.
            # If stuck, try swapping.
            # But 1-opt should exhaust all possible additions.
            # If no edge can be added, we are at a local optimum.
            # Try 2-opt: remove edge e1, add edge e2.
            # This keeps edge count same. Useful to escape local optima for 1-opt?
            # No, 1-opt only adds. If we can't add, we need to remove to allow adding more.
            # So 2-opt: remove e1, add e2, add e3? (Net +1)
            # Or remove e1, add e2 (Net 0) to reach a different local optimum.
            
            # Let's try a few random 2-swaps (remove 1, add 1)
            existing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current[i] >> j) & 1:
                        existing.append((i, j))
                        
            if not existing or not missing:
                break
                
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                e1 = random.choice(existing)
                e2 = random.choice(missing)
                
                u1, v1 = e1
                u2, v2 = e2
                
                remove_edge(current, u1, v1)
                if can_add_edge(current, u2, v2):
                    add_edge(current, u2, v2)
                    # Update lists
                    existing.remove(e1)
                    existing.append(e2)
                    missing.remove(e2)
                    missing.append(e1)
                    improved = True
                    break
                else:
                    add_edge(current, u1, v1)
                    
        return current, current_edges

    best_adj = adj[:]
    best_edges = count_edges(best_adj)
    
    # Run local search
    adj, edges = local_search(adj, deadline)
    if edges > best_edges:
        best_edges = edges
        best_adj = adj[:]
        
    # Convert to list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
