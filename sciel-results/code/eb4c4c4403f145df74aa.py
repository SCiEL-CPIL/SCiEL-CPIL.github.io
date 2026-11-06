import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    q3_edges = set()
    for i in range(8):
        for d in [1, 2, 4]:
            j = i ^ d
            q3_edges.add(frozenset([i, j]))
    
    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_q3(adj):
        for verts in itertools.combinations(range(n), 8):
            verts_set = set(verts)
            degs = [sum(1 for u in verts if u != v and u in adj[v]) for v in verts]
            if any(d != 3 for d in degs):
                continue
            
            color = {}
            bipartite = True
            for v in verts:
                if v not in color:
                    color[v] = 0
                    stack = [v]
                    while stack and bipartite:
                        cur = stack.pop()
                        for nb in adj[cur]:
                            if nb in verts_set:
                                if nb not in color:
                                    color[nb] = 1 - color[cur]
                                    stack.append(nb)
                                elif color[nb] == color[cur]:
                                    bipartite = False
                                    break
                if not bipartite:
                    break
            if not bipartite:
                continue
            
            part_a = [v for v in verts if color[v] == 0]
            part_b = [v for v in verts if color[v] == 1]
            if len(part_a) != 4 or len(part_b) != 4:
                continue
            
            q3_a = [0, 3, 5, 6]
            q3_b = [1, 2, 4, 7]
            
            for pa_perm in itertools.permutations(range(4)):
                ma = {part_a[i]: q3_a[pa_perm[i]] for i in range(4)}
                for pb_perm in itertools.permutations(range(4)):
                    mb = {part_b[j]: q3_b[pb_perm[j]] for j in range(4)}
                    valid = True
                    for i in range(4):
                        for j in range(4):
                            va, vb = part_a[i], part_b[j]
                            if (vb in adj[va]) != (frozenset([ma[va], mb[vb]]) in q3_edges):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return True
        return False
    
    # Try distance-2 graph
    dist2_edges = [(i, j) for i in range(n) for j in range(i+1, n) if bin(i ^ j).count('1') == 2]
    adj = build_adj(dist2_edges)
    if not has_q3(adj):
        return [[u, v] for u, v in dist2_edges]
    
    best_count = 0
    best_edges = []
    
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        edges = [(u, v) for u, v in all_edges if random.random() < 0.55]
        adj = build_adj(edges)
        if not has_q3(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
    
    if best_edges:
        adj = build_adj(best_edges)
        edge_set = set(best_edges)
        while time.time() < deadline:
            missing = [e for e in all_edges if e not in edge_set]
            if not missing:
                break
            random.shuffle(missing)
            improved = False
            for u, v in missing[:50]:
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                if not has_q3(adj):
                    improved = True
                    break
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.discard((u, v))
            if not improved:
                break
    
    if best_count > 0:
        return [[u, v] for u, v in best_edges]
    
    return [[a, b] for a in range(3) for b in range(3, 16)]
