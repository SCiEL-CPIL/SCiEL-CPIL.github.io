import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 90s)
    
    n = 512  # numbers 1..512
    
    # Precompute compatibility as bitsets
    # adj[i] is a Python int bitset where bit j is set if num_i and num_j are compatible
    adj = [0] * n
    for i in range(n):
        num_i = i + 1
        bits = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= n:
                bits |= (1 << j)
        adj[i] = bits
    
    # Popcount helper
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy coloring for upper bound on clique size in a subgraph
    # vertices is a bitset of candidate vertices
    # Returns a list of color classes (each a bitset)
    def greedy_color(vertices):
        color_classes = []
        remaining = vertices
        while remaining:
            # Pick a vertex with minimum degree in remaining (or just lowest index for speed)
            # For speed, pick the lowest-set bit
            v = remaining & (-remaining)
            v_idx = v.bit_length() - 1
            color_classes.append(v)
            # Remove v and all its neighbors from remaining
            remaining &= ~(v | adj[v_idx])
        return len(color_classes)
    
    # Better coloring: process vertices in order, assign smallest available color
    def greedy_color_bound(vertices):
        """Return upper bound on clique size using greedy coloring."""
        # Extract vertex list
        verts = []
        v = vertices
        while v:
            lb = v & (-v)
            idx = lb.bit_length() - 1
            verts.append(idx)
            v ^= lb
        
        if not verts:
            return 0
        
        # Greedy coloring: assign colors to vertices
        color_of = {}
        color_count = 0
        # Process in order
        for v in verts:
            # Find smallest color not used by neighbors of v in verts
            used_colors = set()
            for u in verts:
                if u == v:
                    continue
                if (adj[v] >> u) & 1:
                    if u in color_of:
                        used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            color_count = max(color_count, c + 1)
        
        return color_count
    
    # Branch and bound for maximum clique
    best_clique = []
    
    def search(candidates, current_clique, depth):
        nonlocal best_clique
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > len(best_clique):
                best_clique = list(current_clique)
            return
        
        # Upper bound: current_clique size + color bound on candidates
        cb = greedy_color_bound(candidates)
        if len(current_clique) + cb <= len(best_clique):
            return
        
        # Pick vertex: try to find one that maximizes potential
        # Use the vertex with highest degree in candidates for better pruning
        # Or just pick lowest index for simplicity and determinism
        v = candidates & (-candidates)
        v_idx = v.bit_length() - 1
        
        # Include v
        new_candidates = candidates & adj[v_idx]
        search(new_candidates, current_clique + [v_idx + 1], depth + 1)
        
        # Exclude v
        candidates ^= v
        search(candidates, current_clique, depth)
    
    # Start with all vertices
    all_vertices = (1 << n) - 1
    search(all_vertices, [], 0)
    
    if not best_clique:
        # Fallback: return the simple seed
        from math import isqrt
        return list(range(1, isqrt(512) + 1))
    
    return sorted(best_clique)
