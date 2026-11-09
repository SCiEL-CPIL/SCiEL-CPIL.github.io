# EVOLVE-BLOCK-START
import time
import random
from random import Random

_DEADLINE = 145.0  # safety margin before 150s kill

def _gen_trees(n=20, seed=611):
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
        edges.append((u, w))
        trees.append(edges)
    return trees

def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embedding_order(adj, k):
    """Return vertex order: root first, then BFS order (parents before children)."""
    if k <= 1:
        return list(range(k))
    root = 0
    visited = [False] * k
    order = [root]
    visited[root] = True
    frontier = [root]
    while frontier:
        next_frontier = []
        for node in frontier:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    order.append(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    for v in range(k):
        if not visited[v]:
            order.append(v)
    return order

def construct():
    t0 = time.time()
    N = 20
    trees = _gen_trees(N, 611)
    
    # Precompute tree degrees and adjacency
    tree_info = []
    for k in range(1, N + 1):
        edges = trees[k - 1]
        deg = [0] * k
        adj = [set() for _ in range(k)]
        adj_list = [[] for _ in range(k)]
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].add(b)
            adj[b].add(a)
            adj_list[a].append(b)
            adj_list[b].append(a)
        order = _embedding_order(adj_list, k)
        tree_info.append((k, edges, deg, adj, adj_list, order))
    
    # Embed in decreasing order of size (hardest first)
    order = list(range(N - 1, -1, -1))
    
    # Host state
    used_edges = set()
    used_neighbors = [set() for _ in range(N)]
    
    result = [None] * N
    
    def avail_degree(v):
        return N - 1 - len(used_neighbors[v])
    
    def edge_free(u, v):
        if u > v:
            u, v = v, u
        return (u, v) not in used_edges
    
    def mark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.add((u, v))
        used_neighbors[u].add(v)
        used_neighbors[v].add(u)
    
    def unmark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.discard((u, v))
        used_neighbors[u].discard(v)
        used_neighbors[v].discard(u)
    
    def embed_tree(k_idx, rng):
        """Try to embed tree k_idx (0-indexed, tree size k_idx+1). Returns mapping or None."""
        k = k_idx + 1
        edges, deg, adj, adj_list, v_order = tree_info[k_idx][1], tree_info[k_idx][2], tree_info[k_idx][3], tree_info[k_idx][4], tree_info[k_idx][5]
        
        if k == 1:
            for h in range(N):
                result[k_idx] = [h]
                return True
            return False
        
        if k == 2:
            for u in range(N):
                for v in range(u + 1, N):
                    if edge_free(u, v):
                        mark_edge(u, v)
                        result[k_idx] = [u, v]
                        return True
            return False
        
        # Improved backtracking with forward checking
        mapping = [None] * k
        assigned = [False] * k
        used_host_set = set()
        
        def forward_check(pos):
            """Check that each unassigned vertex has at least one valid host."""
            for i in range(pos, k):
                tv = v_order[i]
                d_needed = deg[tv]
                placed_neighbors = [nb for nb in adj_list[tv] if assigned[nb]]
                
                has_candidate = False
                for h in range(N):
                    if h in used_host_set:
                        continue
                    if avail_degree(h) < d_needed:
                        continue
                    ok = True
                    for nb in placed_neighbors:
                        if not edge_free(h, mapping[nb]):
                            ok = False
                            break
                    if ok:
                        has_candidate = True
                        break
                if not has_candidate:
                    return False
            return True
        
        def backtrack(pos):
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == k:
                return True
            
            tv = v_order[pos]
            d_needed = deg[tv]
            placed_neighbors = [nb for nb in adj_list[tv] if assigned[nb]]
            
            candidates = []
            for h in range(N):
                if h in used_host_set:
                    continue
                if avail_degree(h) < d_needed:
                    continue
                ok = True
                for nb in placed_neighbors:
                    if not edge_free(h, mapping[nb]):
                        ok = False
                        break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort by scarcity: prefer hosts with fewer remaining options
            candidates.sort(key=lambda h: (avail_degree(h), rng.random()))
            max_cands = min(len(candidates), 12)
            
            for h in candidates[:max_cands]:
                mapping[tv] = h
                assigned[tv] = True
                used_host_set.add(h)
                
                for nb in placed_neighbors:
                    mark_edge(h, mapping[nb])
                
                if not forward_check(pos + 1):
                    for nb in placed_neighbors:
                        unmark_edge(h, mapping[nb])
                    mapping[tv] = None
                    assigned[tv] = False
                    used_host_set.discard(h)
                    continue
                
                if backtrack(pos + 1):
                    return True
                
                for nb in placed_neighbors:
                    unmark_edge(h, mapping[nb])
                
                mapping[tv] = None
                assigned[tv] = False
                used_host_set.discard(h)
            
            return False
        
        if backtrack(0):
            result[k_idx] = [int(mapping[i]) for i in range(k)]
            return True
        return False
    
    max_restarts = 80
    best_result = None
    best_count = 0
    
    for restart in range(max_restarts):
        if time.time() - t0 > _DEADLINE:
            break
        
        used_edges.clear()
        for i in range(N):
            used_neighbors[i].clear()
        result = [None] * N
        
        rng = random.Random(611 + restart * 7919)
        
        count = 0
        for k_idx in order:
            if time.time() - t0 > _DEADLINE:
                break
            if embed_tree(k_idx, rng):
                count += 1
            else:
                break
        
        # Polish pass: try to embed remaining trees in different order
        if count < N and time.time() - t0 < _DEADLINE - 5:
            remaining = [k_idx for k_idx in range(N) if result[k_idx] is None]
            # Try smallest first for polish
            remaining.sort(key=lambda x: x + 1)
            for k_idx in remaining:
                if time.time() - t0 > _DEADLINE:
                    break
                if embed_tree(k_idx, rng):
                    count += 1
        
        if count > best_count:
            best_count = count
            best_result = [r[:] if r is not None else None for r in result]
        
        if best_count == N:
            break
    
    if best_result is not None:
        for i in range(N):
            if best_result[i] is not None:
                best_result[i] = [int(x) for x in best_result[i]]
        return best_result
    
    return [None] * N
# EVOLVE-BLOCK-END
