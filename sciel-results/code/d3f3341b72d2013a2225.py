"""Q3-free graph on 16 vertices - maximize edges."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n = 16
    # Precompute all Q3 embeddings as edge sets
    # A Q3 on 8 vertices: label them 0-7 as binary
    # Edges: pairs differing in exactly one bit
    q3_edge_patterns = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edge_patterns.append((i, j))
    
    # All possible vertex assignments of 8 distinct vertices from 0..15
    # C(16,8) = 12870, times 8! = too many
    # Instead, we'll detect Q3 on-the-fly
    
    def has_q3(adj):
        """Check if graph contains a Q3 subgraph."""
        for v in range(n):
            nbrs = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
            if len(nbrs) < 3:
                continue
            # Find independent triples among neighbors
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    if (adj[a] >> b) & 1:
                        continue
                    for k in range(j+1, len(nbrs)):
                        c = nbrs[k]
                        if (adj[a] >> c) & 1 or (adj[b] >> c) & 1:
                            continue
                        # v connected to a,b,c; a,b,c independent
                        # Find a' (adj to b,c not a,v), b' (adj to a,c not b,v), c' (adj to a,b not c,v)
                        for ap in range(n):
                            if ap in (v, a, b, c):
                                continue
                            if not ((adj[ap] >> b) & 1) or not ((adj[ap] >> c) & 1):
                                continue
                            if (adj[ap] >> a) & 1:
                                continue
                            for bp in range(n):
                                if bp in (v, a, b, c, ap):
                                    continue
                                if not ((adj[bp] >> a) & 1) or not ((adj[bp] >> c) & 1):
                                    continue
                                if (adj[bp] >> b) & 1:
                                    continue
                                for cp in range(n):
                                    if cp in (v, a, b, c, ap, bp):
                                        continue
                                    if not ((adj[cp] >> a) & 1) or not ((adj[cp] >> b) & 1):
                                        continue
                                    if (adj[cp] >> c) & 1:
                                        continue
                                    for vp in range(n):
                                        if vp in (v, a, b, c, ap, bp, cp):
                                            continue
                                        if not ((adj[vp] >> ap) & 1) or not ((adj[vp] >> bp) & 1) or not ((adj[vp] >> cp) & 1):
                                            continue
                                        return True
        return False
    
    # Start with a known Q3-free graph: K3,13 (39 edges)
    # Or try to build up from there
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    current_edges = 39
    best_edges = 39
    best_adj = adj[:]
    
    # Try to add edges
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Try adding a random non-edge
        added = False
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j or ((adj[i] >> j) & 1):
                continue
            # Try adding
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            current_edges += 1
            if not has_q3(adj):
                best_edges = current_edges
                best_adj = adj[:]
                added = True
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                current_edges -= 1
            if time.time() > deadline - 5:
                break
        if not added:
            # Try removing an edge and re-adding different ones
            # Or just break
            break
    
    # Final greedy add
    final_adj = best_adj[:]
    final_edges = best_edges
    for i in range(n):
        if time.time() > deadline - 2:
            break
        for j in range(i+1, n):
            if time.time() > deadline - 2:
                break
            if not ((final_adj[i] >> j) & 1):
                final_adj[i] |= (1 << j)
                final_adj[j] |= (1 << i)
                final_edges += 1
                if has_q3(final_adj):
                    final_adj[i] &= ~(1 << j)
                    final_adj[j] &= ~(1 << i)
                    final_edges -= 1
    
    return [[i, j] for i in range(n) for j in range(i+1, n) if (final_adj[i] >> j) & 1]
