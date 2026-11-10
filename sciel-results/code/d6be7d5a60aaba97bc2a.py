import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Vertices 0..14
    # Base graph: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
            
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_path_3(adj, s, t):
        # Check if there is a path s-u-v-t in adj
        if s == t:
            return False
        for u in adj[s]:
            if u == t:
                continue
            for v in adj[u]:
                if v == s or v == t:
                    continue
                if t in adj[v]:
                    return True
        return False

    def is_valid_incremental(curr_deg, curr_adj, new_edge):
        u, v = new_edge
        new_deg = curr_deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        groups = {}
        for i in range(n):
            d = new_deg[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
            
        new_adj = [s.copy() for s in curr_adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        
        for deg_val, verts in groups.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s, t = verts[i], verts[j]
                    if has_path_3(new_adj, s, t):
                        return False
        return True

    def is_valid_full(edges, deg, adj):
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Initial state
    best_edges = set(base_edges)
    best_deg = get_degrees(best_edges)
    best_adj = build_adj(best_edges)
    best_count = len(best_edges)
    
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append(tuple(sorted((A[i], A[j]))))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append(tuple(sorted((B[i], B[j]))))
            
    # Greedy Addition
    curr = set(base_edges)
    curr_deg = get_degrees(curr)
    curr_adj = build_adj(curr)
    
    random.shuffle(candidates)
    
    for c in candidates:
        if c in curr:
            continue
        if is_valid_incremental(curr_deg, curr_adj, c):
            u, v = c
            curr.add(c)
            curr_deg[u] += 1
            curr_deg[v] += 1
            curr_adj[u].add(v)
            curr_adj[v].add(u)
            
    if len(curr) > best_count:
        best_edges = curr
        best_deg = curr_deg
        best_adj = curr_adj
        best_count = len(curr)
        
    # Local Search: Add one, Remove one
    max_iter = 2000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to add an edge
        for add_e in candidates:
            if add_e in best_edges:
                continue
            if is_valid_incremental(best_deg, best_adj, add_e):
                u, v = add_e
                best_edges.add(add_e)
                best_deg[u] += 1
                best_deg[v] += 1
                best_adj[u].add(v)
                best_adj[v].add(u)
                best_count += 1
                improved = True
                break
        if improved:
            continue
            
        if time.time() > deadline - 5:
            break
            
        # Pick a random candidate to add
        add_e = random.choice(candidates)
        if add_e in best_edges:
            continue
            
        # Pick a random edge to remove
        rem_e = random.choice(list(best_edges))
        
        # Check if (best_edges - rem_e + add_e) is valid
        test_edges = (best_edges | {add_e}) - {rem_e}
        test_deg = get_degrees(test_edges)
        test_adj = build_adj(test_edges)
        if is_valid_full(test_edges, test_deg, test_adj):
            best_edges = test_edges
            best_deg = test_deg
            best_adj = test_adj
            
    result = [list(e) for e in best_edges]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    return result
