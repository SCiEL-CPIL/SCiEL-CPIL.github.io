"""Packing trees T1..T20 into K20 using a fresh randomized backtracking approach."""
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 110.0
    n = 20
    trees = instance_trees(n, 611)
    
    used_edges = set()
    result = [None] * n
    
    # Precompute tree adjacency
    tree_adj = []
    for edges in trees:
        k = len(edges) + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: larger trees first
    order = sorted(range(n), key=lambda i: len(trees[i]), reverse=True)
    
    for k_idx in order:
        if time.time() > deadline:
            break
        
        k = k_idx + 1
        if k == 1:
            result[k_idx] = [0]
            continue
        
        adj = tree_adj[k_idx]
        # Vertices ordered by degree descending
        verts = sorted(range(k), key=lambda v: -len(adj[v]))
        
        success = False
        # Try multiple random attempts
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            host = [-1] * k
            used_hosts = set()
            
            def can_assign(v, h):
                """Check if host h can be assigned to tree vertex v."""
                if h in used_hosts:
                    return False
                for u in adj[v]:
                    if host[u] != -1:
                        e = (min(host[u], h), max(host[u], h))
                        if e in used_edges:
                            return False
                return True
            
            def has_available_host(w, tentative_host_v=None, tentative_h_v=None):
                """Check if unassigned tree vertex w has at least one available host."""
                for hw in range(n):
                    if hw in used_hosts:
                        continue
                    if tentative_h_v is not None and hw == tentative_h_v:
                        continue
                    ok = True
                    for u in adj[w]:
                        if u == w:
                            continue
                        if host[u] != -1:
                            hu = host[u]
                        elif u == tentative_host_v:
                            hu = tentative_h_v
                        else:
                            continue
                        e = (min(hu, hw), max(hu, hw))
                        if e in used_edges:
                            ok = False
                            break
                    if ok:
                        return True
                return False
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(verts):
                    return True
                
                v = verts[idx]
                
                # Get candidates
                cands = []
                for h in range(n):
                    if can_assign(v, h):
                        cands.append(h)
                
                if not cands:
                    return False
                
                # Shuffle for randomness
                random.shuffle(cands)
                
                for h in cands:
                    # Forward check: ensure each unassigned tree vertex still has options
                    fc_ok = True
                    for w in range(k):
                        if host[w] == -1 and w != v:
                            if not has_available_host(w, v, h):
                                fc_ok = False
                                break
                    
                    if not fc_ok:
                        continue
                    
                    # Assign
                    host[v] = h
                    used_hosts.add(h)
                    # Add edges to used_edges temporarily
                    added = []
                    for u in adj[v]:
                        if host[u] != -1:
                            e = (min(host[u], h), max(host[u], h))
                            used_edges.add(e)
                            added.append(e)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    for e in added:
                        used_edges.discard(e)
                    host[v] = -1
                    used_hosts.discard(h)
                
                return False
            
            if backtrack(0):
                # Success: add remaining edges to used_edges
                for u, v in trees[k_idx]:
                    a, b = host[u], host[v]
                    used_edges.add((min(a, b), max(a, b)))
                result[k_idx] = [int(x) for x in host]
                success = True
                break
        
        if not success:
            result[k_idx] = None
    
    return result

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used = set()
    ok = True
    total = 0
    for k in range(1, 21):
        r = res[k - 1]
        if r is None:
            continue
        if len(r) != k or len(set(r)) != k:
            ok = False
        for v in r:
            if not isinstance(v, int) or v < 0 or v > 19:
                ok = False
        for (u, v) in trees[k - 1]:
            key = (min(r[u], r[v]), max(r[u], r[v]))
            if key in used:
                ok = False
            used.add(key)
        total += len(trees[k - 1])
    print(f"Valid: {ok}, Total edges: {total} / 190")
