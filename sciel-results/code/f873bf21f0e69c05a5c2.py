"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    
    # Precompute all K20 edges
    K20_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            K20_edges.add((u, v))
    
    def embed_tree_greedy_backtrack(tree_edges, k, used_edges, deadline, rng, max_backtrack=10000):
        """
        Embed a tree with k vertices into K20 avoiding used_edges.
        Uses greedy assignment with limited backtracking.
        Returns a list of k distinct host vertices, or None.
        """
        if k == 1:
            return [0]
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None
        
        # Build adjacency list
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a good root: vertex with highest degree
        root = max(range(k), key=lambda x: len(adj[x]))
        
        # BFS order from root
        visited = [False] * k
        order = [root]
        visited[root] = True
        queue = [root]
        parent = [-1] * k
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    parent[v] = u
                    queue.append(v)
        
        host = [None] * k
        used_hosts = set()
        backtrack_count = [0]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if backtrack_count[0] > max_backtrack:
                return False
            if idx == len(order):
                return True
            
            v = order[idx]
            p = parent[v]
            hp = host[p]
            
            # Collect candidates
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                # Check edge to parent is available
                edge_to_parent = (min(hp, h), max(hp, h))
                if edge_to_parent in used_edges:
                    continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            # Score candidates: prefer those with more available edges to unassigned children
            def score_candidate(h):
                score = 0
                # Count available neighbors that could serve as children
                for child in adj[v]:
                    if child != p:
                        edge = (min(h, child), max(h, child))
                        # This is a proxy; we don't know child's host yet
                        pass
                # Simpler: prefer vertices with fewer used edges (more flexibility)
                used_incident = 0
                for nb in range(20):
                    if nb != h:
                        e = (min(h, nb), max(h, nb))
                        if e in used_edges:
                            used_incident += 1
                score = 19 - used_incident
                return score
            
            scored = [(score_candidate(h), h) for h in candidates]
            scored.sort(key=lambda x: -x[0])
            
            # Add randomization for ties
            for i in range(len(scored) - 1, 0, -1):
                if rng.random() < 0.2:
                    scored[i], scored[i-1] = scored[i-1], scored[i]
            
            for _, h in scored:
                backtrack_count[0] += 1
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
            
            return False
        
        # Try with different rootings
        for attempt in range(10):
            if time.time() > deadline:
                break
            host = [None] * k
            used_hosts = set()
            backtrack_count[0] = 0
            
            if backtrack(0):
                return host
        return None
    
    def embed_t20_fast(tree_edges, deadline, rng):
        """
        Specialized fast embedder for T20 (spanning tree, uses all 20 vertices).
        Uses degree-constrained greedy with limited backtracking.
        """
        k = 20
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find centroid
        deg = [len(adj[i]) for i in range(k)]
        leaves = [i for i in range(k) if deg[i] == 1]
        remaining = k
        while remaining > 2:
            new_leaves = []
            for leaf in leaves:
                remaining -= 1
                for nb in adj[leaf]:
                    deg[nb] -= 1
                    if deg[nb] == 1 and remaining > 1:
                        new_leaves.append(nb)
            leaves = new_leaves
        root = leaves[0] if leaves else 0
        
        # BFS order
        visited = [False] * k
        order = [root]
        visited[root] = True
        queue = [root]
        parent = [-1] * k
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    parent[v] = u
                    queue.append(v)
        
        # Greedy assignment with backtracking
        host = [None] * k
        used_hosts = set()
        used_edges_local = set()
        backtrack_count = [0]
        max_bt = 50000
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if backtrack_count[0] > max_bt:
                return False
            if idx == len(order):
                return True
            
            v = order[idx]
            p = parent[v]
            hp = host[p]
            
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                edge_to_parent = (min(hp, h), max(hp, h))
                if edge_to_parent in used_edges_local:
                    continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            # Score: prefer vertices that leave more options for children
            def score(h):
                # Count how many of h's neighbors are still available
                avail = 0
                for nb in range(20):
                    if nb != h and nb not in used_hosts:
                        e = (min(h, nb), max(h, nb))
                        if e not in used_edges_local:
                            avail += 1
                return avail
            
            scored = [(score(h), h) for h in candidates]
            scored.sort(key=lambda x: -x[0])
            
            # Randomize ties slightly
            for i in range(len(scored) - 1, 0, -1):
                if rng.random() < 0.15:
                    scored[i], scored[i-1] = scored[i-1], scored[i]
            
            for _, h in scored:
                backtrack_count[0] += 1
                host[v] = h
                used_hosts.add(h)
                # Add edge to parent to local used set
                edge_to_parent = (min(hp, h), max(hp, h))
                used_edges_local.add(edge_to_parent)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
                used_edges_local.discard(edge_to_parent)
            
            return False
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            host = [None] * k
            used_hosts = set()
            used_edges_local = set()
            backtrack_count[0] = 0
            
            if backtrack(0):
                return host
        return None
    
    best_score = 0
    best_embeddings = [None] * 20
    
    # T1: single vertex
    best_embeddings[0] = [0]
    
    # Strategy: Try embedding T20 first, then T19, T18, ..., T2
    for restart in range(80):
        if time.time() > deadline - 3:
            break
        
        rng = Random(7777 + restart * 137)
        used_edges = set()
        embeddings = [None] * 20
        embeddings[0] = [0]
        
        # Embed T20 first
        res20 = embed_t20_fast(trees[19], deadline, rng)
        if res20 is None:
            continue
        embeddings[19] = res20
        for u, v in trees[19]:
            used_edges.add((min(res20[u], res20[v]), max(res20[u], res20[v])))
        
        # Embed T19 down to T2
        success = True
        for k in range(19, 1, -1):
            if time.time() > deadline - 3:
                success = False
                break
            
            if k <= 8:
                res = embed_tree_greedy_backtrack(trees[k-1], k, used_edges, deadline, rng, max_backtrack=20000)
            else:
                res = embed_tree_greedy_backtrack(trees[k-1], k, used_edges, deadline, rng, max_backtrack=50000)
            
            if res is None:
                success = False
                break
            
            embeddings[k-1] = res
            for u, v in trees[k-1]:
                used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
        
        if success:
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
            
            if best_score == 190:
                break
    
    # Fallback: if T20 wasn't embedded, try to add it
    if best_embeddings[19] is None:
        used_edges = set()
        for k in range(2, 20):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        rng = Random(424242)
        res20 = embed_t20_fast(trees[19], deadline, rng)
        if res20 is not None:
            # Check if edges are disjoint
            valid = True
            for u, v in trees[19]:
                e = (min(res20[u], res20[v]), max(res20[u], res20[v]))
                if e in used_edges:
                    valid = False
                    break
            if valid:
                best_embeddings[19] = res20
                best_score += 19
    
    # Fallback: try to add T2 if missing
    if best_embeddings[1] is None:
        used_edges = set()
        for k in range(2, 21):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        for u in range(20):
            for v in range(u+1, 20):
                if (u, v) not in used_edges:
                    best_embeddings[1] = [u, v]
                    best_score += 1
                    break
            if best_embeddings[1] is not None:
                break
    
    # Final validation and cleanup
    if best_embeddings[0] is None:
        best_embeddings[0] = [0]
        
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None
            else:
                if len(set(best_embeddings[i])) != i + 1:
                    best_embeddings[i] = None
                else:
                    # Ensure all are in 0..19 and are Python ints
                    for j in range(len(best_embeddings[i])):
                        val = best_embeddings[i][j]
                        if not isinstance(val, int) or isinstance(val, bool) or val < 0 or val > 19:
                            best_embeddings[i] = None
                            break
                        best_embeddings[i][j] = int(val)
    
    # Final edge-disjointness check
    all_used = set()
    for i in range(20):
        if best_embeddings[i] is None:
            continue
        for u, v in trees[i]:
            e = (min(best_embeddings[i][u], best_embeddings[i][v]), 
                 max(best_embeddings[i][u], best_embeddings[i][v]))
            if e in all_used:
                best_embeddings[i] = None
            else:
                all_used.add(e)
    
    return best_embeddings

# EVOLVE-BLOCK-END
