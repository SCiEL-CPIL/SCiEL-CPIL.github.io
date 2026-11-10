"""
Combined approach:
1. Use bitmask representation for fast triangle-freeness and 3-colorability checks
2. Start from Grötzsch graph and use blow-up distributions (structural idea from inspiration)
3. Use greedy edge addition with random restarts (refinement from current)
4. Use adaptive Maximum Independent Set extension for densification
"""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    n = 16
    
    # Bitmask helpers
    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                common = adj[v] & adj[u]
                if common & ~(1 << u) & ~(1 << v):
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        # Check if adding u-v creates a triangle
        common = adj[u] & adj[v]
        if common & ~(1 << u) & ~(1 << v):
            return False
        return True
    
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def bt(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            mask = adj[v]
            while mask:
                lsb = mask & -mask
                u = lsb.bit_length() - 1
                mask ^= lsb
                if colors[u] != -1:
                    used |= 1 << colors[u]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if bt(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return bt(0)
    
    def edge_count(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Build Grötzsch graph (11 vertices)
    def build_groetzsch_adj():
        adj = [0] * n
        # Pentagon 0-4
        for i in range(5):
            u, v = i, (i + 1) % 5
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        # Mycielski: vertices 5-9
        for i in range(5):
            u, v = i, (i + 1) % 5
            adj[u] |= 1 << (5 + v)
            adj[5 + v] |= 1 << u
            adj[v] |= 1 << (5 + u)
            adj[5 + u] |= 1 << v
        # Vertex 10 connected to 5-9
        for u in range(5):
            adj[5 + u] |= 1 << 10
            adj[10] |= 1 << (5 + u)
        return adj
    
    # Maximum Independent Set approximation using greedy
    def max_independent_set(adj, vertices):
        """Find a large independent set among given vertices"""
        remaining = set(vertices)
        result = []
        while remaining:
            # Pick vertex with minimum degree in induced subgraph
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and adj[v] & (1 << u):
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            result.append(best_v)
            remaining.remove(best_v)
            # Remove neighbors of best_v
            mask = adj[best_v]
            to_remove = set()
            for u in remaining:
                if mask & (1 << u):
                    to_remove.add(u)
            remaining -= to_remove
        return result
    
    # Phase 1: Generate blow-up distributions
    def generate_distributions(num_extra, num_original):
        dists = []
        def rec(rem, idx, current):
            if idx == num_original - 1:
                current.append(rem)
                dists.append(tuple(current))
                current.pop()
                return
            for i in range(rem + 1):
                current.append(i)
                rec(rem - i, idx + 1, current)
                current.pop()
        rec(num_extra, 0, [])
        return dists
    
    num_original = 11
    num_extra = n - num_original
    dists = generate_distributions(num_extra, num_original)
    
    g_adj = build_groetzsch_adj()
    
    best_edges = []
    best_count = 0
    
    # Try all blow-up distributions
    for dist in dists:
        if time.time() > deadline - 30:
            break
        
        sizes = [1 + d for d in dist]
        
        # Map each vertex to its original Grötzsch vertex
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        adj = [0] * n
        edges = []
        
        # Add edges between blown-up vertices
        for i in range(num_original):
            for j in range(i + 1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        # Check triangle-free
        if not is_triangle_free(adj):
            continue
        
        # Check 3-colorability
        if is_3_colorable(adj):
            continue
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = list(edges)
    
    # If no valid graph found, use base Grötzsch
    if not best_edges:
        adj = build_groetzsch_adj()
        best_edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    best_edges.append([u, v])
        best_count = edge_count(adj)
    else:
        adj = build_adj(best_edges)
    
    # Phase 2: Greedy edge addition with random restarts
    for restart in range(500):
        if time.time() > deadline - 20:
            break
        
        # Start from current best
        cur_adj = [x for x in adj]
        cur_edges = list(best_edges)
        cur_count = best_count
        
        # Get non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (cur_adj[u] & (1 << v)):
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        improved = False
        for u, v in non_edges:
            if time.time() > deadline - 20:
                break
            
            if not (cur_adj[u] & (1 << v)) and can_add_edge(cur_adj, u, v):
                cur_adj[u] |= 1 << v
                cur_adj[v] |= 1 << u
                cur_edges.append([u, v])
                
                if not is_3_colorable(cur_adj):
                    ec = edge_count(cur_adj)
                    if ec > best_count:
                        best_count = ec
                        best_edges = list(cur_edges)
                        adj = [x for x in cur_adj]
                        improved = True
                else:
                    cur_adj[u] ^= 1 << v
                    cur_adj[v] ^= 1 << u
                    cur_edges.pop()
        
        if not improved and restart > 100:
            # Try perturbation: remove a random edge and re-add
            if best_edges:
                u, v = random.choice(best_edges)
                cur_adj = [x for x in adj]
                cur_adj[u] ^= 1 << v
                cur_adj[v] ^= 1 << u
                cur_edges = list(best_edges)
                cur_edges.remove([u, v])
                
                non_edges2 = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if not (cur_adj[a] & (1 << b)):
                            non_edges2.append((a, b))
                random.shuffle(non_edges2)
                
                for a, b in non_edges2:
                    if time.time() > deadline - 20:
                        break
                    if not (cur_adj[a] & (1 << b)) and can_add_edge(cur_adj, a, b):
                        cur_adj[a] |= 1 << b
                        cur_adj[b] |= 1 << a
                        cur_edges.append([a, b])
                        
                        if not is_3_colorable(cur_adj):
                            ec = edge_count(cur_adj)
                            if ec > best_count:
                                best_count = ec
                                best_edges = list(cur_edges)
                                adj = [x for x in cur_adj]
                        else:
                            cur_adj[a] ^= 1 << b
                            cur_adj[b] ^= 1 << a
                            cur_edges.pop()
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                result.append([u, v])
    
    return result
