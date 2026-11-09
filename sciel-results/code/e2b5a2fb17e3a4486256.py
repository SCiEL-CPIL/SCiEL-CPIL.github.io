"""Packing trees T1..T20 into K20 using edge-coloring decomposition and backtracking."""
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

def _build_factorization(n=20):
    """Build a 1-factorization of K_n. Returns list of n-1 perfect matchings."""
    if n % 2 == 1:
        # For odd n, K_n has (n-1)/2 perfect matchings in a near-1-factorization
        # But K20 is even, so we get 19 perfect matchings
        pass
    # Standard 1-factorization for K_n where n is even
    # Vertices 0..n-1, plus a dummy vertex n for the construction
    # Actually for K_n with n even, we use the standard construction
    # Label vertices 0, 1, ..., n-2, and infinity (n-1)
    # For K_20: vertices 0..19
    # Use the standard round-robin tournament scheduling
    
    factorization = []
    # For K_n with n even, n-1 perfect matchings
    # Standard construction: fix vertex n-1, rotate others
    fixed = n - 1
    others = list(range(n - 1))
    
    for round_idx in range(n - 1):
        matching = []
        # fixed vertex is matched with others[0]
        matching.append((fixed, others[0]))
        for i in range(1, (n - 1) // 2 + 1):
            if i == n - 1 - i:
                continue
            matching.append((others[i], others[n - 1 - i]))
        factorization.append(matching)
        # Rotate: move others[0] to end
        others = others[1:] + [others[0]]
    
    return factorization

def _embed_tree_with_coloring(tree_edges, tree_k, factorization, n, used_matchings, deadline):
    """
    Embed tree with k vertices into K_n such that each edge uses a specific matching.
    tree_edges: list of (u, v) in tree coordinates
    tree_k: number of vertices in tree
    factorization: list of perfect matchings
    n: size of host graph
    used_matchings: set of matching indices already used by previous trees
    deadline: time deadline
    
    Returns: list of host vertices (length k) or None
    """
    if tree_k == 1:
        return [0]
    
    # Build tree adjacency
    adj = [[] for _ in range(tree_k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree descending
    verts = sorted(range(tree_k), key=lambda v: -len(adj[v]))
    
    # For each tree edge, we need to assign it to a matching
    # A matching is a perfect matching of K_n
    # The key constraint: if edge (u,v) in tree is assigned to matching m,
    # then host[u] and host[v] must be paired in matching m
    
    # Strategy: try different assignments of tree edges to available matchings
    available_matchings = [m for m in range(len(factorization)) if m not in used_matchings]
    
    if len(available_matchings) < len(tree_edges):
        # Not enough matchings - but we can reuse matchings if edges don't conflict
        # Actually, multiple tree edges CAN use the same matching as long as
        # the host vertices don't conflict (i.e., a host vertex isn't used twice in same matching)
        # But since it's a tree, each vertex has degree <= something
        # Let's just use backtracking without strict matching assignment first
        pass
    
    # Simpler approach: use backtracking with edge-disjointness check
    # but leverage the factorization for candidate generation
    
    host = [-1] * tree_k
    used_hosts = set()
    used_edges = set()
    
    def get_matching_partners(host_v, matching_idx):
        """Get the partner of host_v in the given matching."""
        for a, b in factorization[matching_idx]:
            if a == host_v:
                return b
            if b == host_v:
                return a
        return None
    
    def can_assign(v, h):
        if h in used_hosts:
            return False
        for u in adj[v]:
            if host[u] != -1:
                e = (min(host[u], h), max(host[u], h))
                if e in used_edges:
                    return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(verts):
            return True
        
        v = verts[idx]
        cands = []
        for h in range(n):
            if can_assign(v, h):
                cands.append(h)
        
        if not cands:
            return False
        
        random.shuffle(cands)
        
        for h in cands:
            host[v] = h
            used_hosts.add(h)
            added = []
            for u in adj[v]:
                if host[u] != -1:
                    e = (min(host[u], h), max(host[u], h))
                    used_edges.add(e)
                    added.append(e)
            
            if backtrack(idx + 1):
                return True
            
            for e in added:
                used_edges.discard(e)
            host[v] = -1
            used_hosts.discard(h)
        
        return False
    
    # Try multiple random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        host = [-1] * tree_k
        used_hosts = set()
        used_edges = set()
        
        # Shuffle vertex order for diversity
        verts_shuffled = list(verts)
        random.shuffle(verts_shuffled)
        
        def backtrack2(idx):
            if time.time() > deadline:
                return False
            if idx == len(verts_shuffled):
                return True
            
            v = verts_shuffled[idx]
            cands = []
            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for u in adj[v]:
                    if host[u] != -1:
                        e = (min(host[u], h), max(host[u], h))
                        if e in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(h)
            
            if not cands:
                return False
            
            random.shuffle(cands)
            
            for h in cands:
                host[v] = h
                used_hosts.add(h)
                added = []
                for u in adj[v]:
                    if host[u] != -1:
                        e = (min(host[u], h), max(host[u], h))
                        used_edges.add(e)
                        added.append(e)
                
                if backtrack2(idx + 1):
                    return True
                
                for e in added:
                    used_edges.discard(e)
                host[v] = -1
                used_hosts.discard(h)
            
            return False
        
        if backtrack2(0):
            return [int(x) for x in host]
    
    return None

def construct():
    deadline = time.time() + 110.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Build 1-factorization of K20
    factorization = _build_factorization(n)
    
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
        verts = sorted(range(k), key=lambda v: -len(adj[v]))
        
        success = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            host = [-1] * k
            used_hosts = set()
            
            def can_assign(v, h):
                if h in used_hosts:
                    return False
                for u in adj[v]:
                    if host[u] != -1:
                        e = (min(host[u], h), max(host[u], h))
                        if e in used_edges:
                            return False
                return True
            
            def has_available_host(w):
                for hw in range(n):
                    if hw in used_hosts:
                        continue
                    ok = True
                    for u in adj[w]:
                        if host[u] != -1:
                            hu = host[u]
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
                cands = []
                for h in range(n):
                    if can_assign(v, h):
                        cands.append(h)
                
                if not cands:
                    return False
                
                random.shuffle(cands)
                
                for h in cands:
                    # Forward check
                    fc_ok = True
                    for w in range(k):
                        if host[w] == -1 and w != v:
                            if not has_available_host(w):
                                fc_ok = False
                                break
                    
                    if not fc_ok:
                        continue
                    
                    host[v] = h
                    used_hosts.add(h)
                    added = []
                    for u in adj[v]:
                        if host[u] != -1:
                            e = (min(host[u], h), max(host[u], h))
                            used_edges.add(e)
                            added.append(e)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    for e in added:
                        used_edges.discard(e)
                    host[v] = -1
                    used_hosts.discard(h)
                
                return False
            
            if backtrack(0):
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
