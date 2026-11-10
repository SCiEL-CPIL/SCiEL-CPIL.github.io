import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110
    N = 16
    
    # Precompute all 8-vertex subsets
    all_8subsets = list(combinations(range(N), 8))
    
    # For each 8-subset, precompute which pairs of vertices (global indices) are in it
    # And for fast checking, we'll use adjacency as 16-bit integers per vertex
    
    # For a graph with adj[i] = bitmask of neighbors of i:
    # For subset S = (v0,...,v7), the induced subgraph is Q3 iff:
    # - 12 edges
    # - all degrees 3
    # - bipartite
    
    def check_q3(subset, adj):
        """adj is list of 16 ints, each a 16-bit mask."""
        v0, v1, v2, v3, v4, v5, v6, v7 = subset
        vs = [v0, v1, v2, v3, v4, v5, v6, v7]
        
        # Build local adjacency (8x8)
        # For speed, compute edge count and degrees
        total_edges = 0
        degs = [0] * 8
        for i, vi in enumerate(vs):
            for j in range(i+1, 8):
                vj = vs[j]
                if (adj[vi] >> vj) & 1:
                    total_edges += 1
                    degs[i] += 1
                    degs[j] += 1
        
        if total_edges != 12:
            return False
        for d in degs:
            if d != 3:
                return False
        
        # Bipartite check
        color = [-1] * 8
        color[0] = 0
        stack = [0]
        while stack:
            u = stack.pop()
            vu = vs[u]
            for v in range(8):
                if v == u:
                    continue
                vv = vs[v]
                if (adj[vu] >> vv) & 1:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
        return True
    
    # Precompute for each 8-subset, the list of pairs within it (for edge lookup)
    # And for each pair (i,j), which 8-subsets contain both i and j
    pair_to_subsets = {}
    for idx, subset in enumerate(all_8subsets):
        for i in range(8):
            for j in range(i+1, 8):
                pair = (min(subset[i], subset[j]), max(subset[i], subset[j]))
                if pair not in pair_to_subsets:
                    pair_to_subsets[pair] = []
                pair_to_subsets[pair].append(idx)
    
    def has_q3_with_edge(edge, adj):
        """Check if adding edge creates any Q3. Only check subsets containing both endpoints."""
        u, v = edge
        pair = (min(u, v), max(u, v))
        if pair not in pair_to_subsets:
            return False
        for idx in pair_to_subsets[pair]:
            if check_q3(all_8subsets[idx], adj):
                return True
        return False
    
    def full_q3_check(adj):
        """Check if graph has any Q3."""
        for subset in all_8subsets:
            if check_q3(subset, adj):
                return True
        return False
    
    best_edges = set()
    best_count = 0
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    start_time = time.time()
    
    for trial in range(30):
        if time.time() > deadline - 3:
            break
        
        # Random start
        density = random.uniform(0.4, 0.8)
        edges = set()
        for pair in all_pairs:
            if random.random() < density:
                edges.add(pair)
        
        adj = [0] * N
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Remove Q3s: find and remove edges
        for iter_count in range(200):
            if time.time() > deadline - 3:
                break
            
            # Find a Q3
            found_q3 = None
            for subset in all_8subsets:
                if check_q3(subset, adj):
                    found_q3 = subset
                    break
            
            if found_q3 is None:
                break
            
            # Remove a random edge from this Q3
            v0, v1, v2, v3, v4, v5, v6, v7 = found_q3
            vs = [v0, v1, v2, v3, v4, v5, v6, v7]
            # Find edges in this Q3
            q3_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj[vs[i]] >> vs[j]) & 1:
                        q3_edges.append((vs[i], vs[j]))
            
            e = random.choice(q3_edges)
            edges.discard(e)
            adj[e[0]] &= ~(1 << e[1])
            adj[e[1]] &= ~(1 << e[0])
        
        # Greedily add edges
        missing = [p for p in all_pairs if p not in edges]
        random.shuffle(missing)
        
        added_any = True
        while added_any:
            if time.time() > deadline - 3:
                break
            added_any = False
            for e in missing:
                if e not in edges:
                    # Try adding
                    edges.add(e)
                    adj[e[0]] |= (1 << e[1])
                    adj[e[1]] |= (1 << e[0])
                    
                    if has_q3_with_edge(e, adj):
                        edges.discard(e)
                        adj[e[0]] &= ~(1 << e[1])
                        adj[e[1]] &= ~(1 << e[0])
                    else:
                        added_any = True
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
    
    # Final polish
    if time.time() < deadline - 2 and best_count > 0:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        missing = [p for p in all_pairs if p not in best_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 1:
                break
            if e not in best_edges:
                best_edges.add(e)
                adj[e[0]] |= (1 << e[1])
                adj[e[1]] |= (1 << e[0])
                
                if has_q3_with_edge(e, adj):
                    best_edges.discard(e)
                    adj[e[0]] &= ~(1 << e[1])
                    adj[e[1]] &= ~(1 << e[0])
    
    return [[u, v] for u, v in best_edges]
