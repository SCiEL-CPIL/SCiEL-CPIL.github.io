# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_index = {}
    for i, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = i

    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v using adjacency matrix."""
        return int(np.dot(adj[u], adj[v]))

    def is_k23_free(adj):
        """Check no pair has >= 3 common neighbors."""
        # common[u,v] = number of w adjacent to both u and v
        # For u != v: (adj @ adj)[u, v] counts common neighbors
        common = adj @ adj
        for u in range(n):
            for v in range(u + 1, n):
                if common[u, v] >= 3:
                    return False
        return True

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def greedy_add(adj):
        """Try to add edges greedily in random order until no more can be added."""
        remaining = [p for p in all_pairs if adj[p[0], p[1]] == 0]
        random.shuffle(remaining)
        for u, v in remaining:
            if adj[u, v] == 1:
                continue
            # Check if adding (u,v) creates K2,3
            # Adding (u,v) means for any w, common neighbors of (u,w) and (v,w) increase
            # Actually: after adding (u,v), for any pair (x,y) not involving u or v,
            # common neighbors don't change. Pairs involving u: (u,z) for z != v
            # common(u,z) increases by 1 if adj[v,z]==1 (v becomes common neighbor of u,z)
            # Similarly pairs involving v.
            # So we need: for all z != u,v:
            #   if adj[v,z]==1: common(u,z) must be <= 1 (so after +1 it's <=2)
            #   if adj[u,z]==1: common(v,z) must be <= 1
            # And for pair (u,v) itself: common(u,v) must be <= 2 (it becomes the new edge,
            #   but common(u,v) is about OTHER vertices being common neighbors, which doesn't
            #   change when we add edge (u,v) itself... wait)
            # Actually common(u,v) = |N(u) ∩ N(v)|. Adding edge (u,v) doesn't change N(u)∩N(v)
            # because u is not in N(u) and v is not in N(v) in a simple graph.
            # Wait, N(u) includes v after adding edge. But common neighbors are vertices w
            # adjacent to BOTH u and v. u itself is not adjacent to u. v itself is not adjacent to v.
            # So adding edge (u,v) doesn't change common(u,v).
            # But it DOES affect common(u,z) for z adjacent to v: v becomes a common neighbor.
            # And common(v,z) for z adjacent to u: u becomes a common neighbor.
            ok = True
            # Check pairs (u, z) where z is adjacent to v
            for z in range(n):
                if z == u or z == v:
                    continue
                if adj[v, z] == 1:
                    if common_neighbors_count(adj, u, z) >= 2:
                        ok = False
                        break
            if ok:
                for z in range(n):
                    if z == u or z == v:
                        continue
                    if adj[u, z] == 1:
                        if common_neighbors_count(adj, v, z) >= 2:
                            ok = False
                            break
            if ok:
                adj[u, v] = 1
                adj[v, u] = 1
        return adj

    def local_search(adj, max_iter=50000):
        """Local search with edge additions and 1-1 swaps."""
        it = 0
        improved = True
        while improved and it < max_iter:
            if time.time() > deadline - 5:
                break
            improved = False
            it += 1
            
            # Try to add a random missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v] == 0]
            if missing:
                random.shuffle(missing)
                for u, v in missing[:200]:  # limit checks per iteration
                    if adj[u, v] == 1:
                        continue
                    ok = True
                    for z in range(n):
                        if z == u or z == v:
                            continue
                        if adj[v, z] == 1 and common_neighbors_count(adj, u, z) >= 2:
                            ok = False
                            break
                        if adj[u, z] == 1 and common_neighbors_count(adj, v, z) >= 2:
                            ok = False
                            break
                    if ok:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        improved = True
                        break
            
            # Try 1-1 swap: remove edge (a,b) and add edge (c,d)
            if not improved:
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v] == 1]
                if len(edges) >= 2:
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        a, b = random.choice(edges)
                        # Find candidate edges to add
                        candidates = [(u, v) for u in range(n) for v in range(u + 1, n) 
                                       if adj[u, v] == 0 and (u, v) != (a, b) and (u, v) != (b, a)]
                        if not candidates:
                            break
                        c, d = random.choice(candidates)
                        # Try the swap
                        adj[a, b] = 0
                        adj[b, a] = 0
                        # Check if we can add (c,d) now
                        ok = True
                        for z in range(n):
                            if z == c or z == d:
                                continue
                            if adj[d, z] == 1 and common_neighbors_count(adj, c, z) >= 2:
                                ok = False
                                break
                            if adj[c, z] == 1 and common_neighbors_count(adj, d, z) >= 2:
                                ok = False
                                break
                        if ok:
                            adj[c, d] = 1
                            adj[d, c] = 1
                            improved = True
                            break
                        else:
                            # Undo
                            adj[a, b] = 1
                            adj[b, a] = 1
        return adj

    def random_init():
        """Create a random sparse graph."""
        adj = np.zeros((n, n), dtype=np.int8)
        # Start with a random subset of edges
        for u, v in all_pairs:
            if random.random() < 0.3:
                adj[u, v] = 1
                adj[v, u] = 1
        return adj

    def structured_init():
        """Create a structured starting graph (e.g., random regular-ish)."""
        adj = np.zeros((n, n), dtype=np.int8)
        # Random 4-regular-ish graph
        for u in range(n):
            neighbors = [v for v in range(n) if v != u]
            random.shuffle(neighbors)
            for v in neighbors[:4]:
                adj[u, v] = 1
                adj[v, u] = 1
        return adj

    best_count = 0
    best_adj = None

    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Start from empty, greedy add
    strategies.append("empty_greedy")
    # Strategy 2: Random init + repair + greedy
    strategies.append("random_greedy")
    # Strategy 3: Structured init + local search
    strategies.append("structured_ls")
    
    for strat in strategies:
        if time.time() > deadline - 10:
            break
        
        if strat == "empty_greedy":
            adj = np.zeros((n, n), dtype=np.int8)
            adj = greedy_add(adj)
            adj = local_search(adj)
        elif strat == "random_greedy":
            adj = random_init()
            # Repair: remove edges that cause violations
            for _ in range(20):
                common = adj @ adj
                violation_found = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if common[u, v] >= 3:
                            # Remove a random edge among common neighbors
                            common_nbs = [w for w in range(n) if adj[u, w] == 1 and adj[v, w] == 1]
                            if common_nbs:
                                w = random.choice(common_nbs)
                                # Remove either (u,w) or (v,w)
                                if random.random() < 0.5:
                                    adj[u, w] = 0
                                    adj[w, u] = 0
                                else:
                                    adj[v, w] = 0
                                    adj[w, v] = 0
                                violation_found = True
                                break
                    if violation_found:
                        break
                if not violation_found:
                    break
            adj = greedy_add(adj)
            adj = local_search(adj)
        elif strat == "structured_ls":
            adj = structured_init()
            adj = greedy_add(adj)
            adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

    # Multiple restarts with local search
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        adj = random_init()
        # Repair
        for _ in range(30):
            common = adj @ adj
            violation_found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] >= 3:
                        common_nbs = [w for w in range(n) if adj[u, w] == 1 and adj[v, w] == 1]
                        if common_nbs:
                            w = random.choice(common_nbs)
                            if random.random() < 0.5:
                                adj[u, w] = 0
                                adj[w, u] = 0
                            else:
                                adj[v, w] = 0
                                adj[w, v] = 0
                            violation_found = True
                            break
                if violation_found:
                    break
            if not violation_found:
                break
        adj = greedy_add(adj)
        adj = local_search(adj, max_iter=30000)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

    # Final verification and extraction
    if best_adj is None:
        best_adj = np.zeros((n, n), dtype=np.int8)

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
