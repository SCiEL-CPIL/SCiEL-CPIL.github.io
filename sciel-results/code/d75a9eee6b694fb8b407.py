import time
from random import Random
import sys

def instance_trees(n=20, seed=611):
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

def _build_tree_info(edges, k):
    if k <= 1:
        return [], {}, [0] * k, [0] * k
    adj = [[] for _ in range(k)]
    deg = [0] * k
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    return edges, adj, deg, [0] * k

def _embed_tree(edges, adj, deg, k, available_edges, host_deg, rng, deadline, max_tries=50):
    """Try to embed tree into available K20 edges. Returns mapping list or None."""
    if k == 1:
        # Find any host vertex with degree >= 0 (always true)
        for _ in range(max_tries):
            h = rng.randint(0, 19)
            return [h]
        return [0]
    
    if k == 2:
        # Just need one available edge
        edge_list = list(available_edges)
        if not edge_list:
            return None
        for _ in range(max_tries):
            u, v = rng.choice(edge_list)
            return [u, v]
        return None
    
    # Order tree vertices by degree descending for better pruning
    tv_order = list(range(k))
    tv_order.sort(key=lambda v: -deg[v])
    
    for attempt in range(max_tries):
        host_map = [-1] * k
        # Assign highest-degree tree vertex first
        tv0 = tv_order[0]
        # Choose host with sufficient degree
        candidates = [h for h in range(20) if host_deg[h] >= deg[tv0]]
        if not candidates:
            return None
        h0 = rng.choice(candidates)
        host_map[tv0] = h0
        
        success = _dfs_assign(tv_order, 1, k, host_map, edges, adj, deg, available_edges, host_deg, rng, deadline)
        if success:
            return host_map
    return None

def _dfs_assign(tv_order, pos, k, host_map, edges, adj, deg, available_edges, host_deg, rng, deadline):
    if time.time() > deadline:
        return False
    if pos == k:
        # Verify all edges are in available_edges
        for u, v in edges:
            hu, hv = host_map[u], host_map[v]
            e = (min(hu, hv), max(hu, hv))
            if e not in available_edges:
                return False
        return True
    
    tv = tv_order[pos]
    used_hosts = set(h for h in host_map if h != -1)
    
    # Get candidate hosts
    cands = [h for h in range(20) if h not in used_hosts and host_deg[h] >= deg[tv]]
    if not cands:
        return False
    
    # Shuffle for diversity
    rng.shuffle(cands)
    
    # Sort by descending available degree for better pruning
    cands.sort(key=lambda h: -host_deg[h])
    
    for h in cands:
        # Check edges to already-assigned neighbors
        valid = True
        for nb in adj[tv]:
            if host_map[nb] != -1:
                hn = host_map[nb]
                e = (min(h, hn), max(h, hn))
                if e not in available_edges:
                    valid = False
                    break
        if not valid:
            continue
        
        host_map[tv] = h
        if _dfs_assign(tv_order, pos + 1, k, host_map, edges, adj, deg, available_edges, host_deg, rng, deadline):
            return True
        host_map[tv] = -1
    
    return False

def _greedy_embed_all(trees, tree_infos, rng, deadline):
    """Embed trees one by one using greedy backtracking. Returns mappings list or None per tree."""
    n = 20
    mappings = [None] * 20
    # Start with full K20
    available = set((i, j) for i in range(n) for j in range(i + 1, n))
    host_deg = [19] * n  # K20 each vertex has degree 19
    
    # Sort trees: largest first for better constraint propagation
    order = list(range(20))
    order.sort(key=lambda i: -(len(trees[i])))
    
    for t_idx in order:
        edges, adj, deg, k = tree_infos[t_idx]
        if k == 1:
            mappings[t_idx] = [rng.randint(0, 19)]
            continue
        if not available:
            break
        
        mapping = _embed_tree(edges, adj, deg, k, available, host_deg, rng, deadline)
        if mapping is None:
            break
        
        mappings[t_idx] = mapping
        # Remove used edges from available
        for u, v in edges:
            hu, hv = mapping[u], mapping[v]
            e = (min(hu, hv), max(hu, hv))
            available.discard(e)
            host_deg[hu] -= 1
            host_deg[hv] -= 1
    
    # Check if all embedded
    if all(m is not None for m in mappings):
        return mappings, available, host_deg
    return mappings, available, host_deg

def _count_embedded_edges(trees, mappings):
    total = 0
    for i, m in enumerate(mappings):
        if m is not None and len(m) > 1:
            total += len(trees[i])
    return total

def _verify_packing(trees, mappings):
    """Verify edge-disjointness and return count of valid embedded trees."""
    if mappings is None:
        return 0
    used_edges = set()
    count = 0
    for i, m in enumerate(mappings):
        if m is None:
            continue
        k = len(m)
        if k == 0:
            continue
        if k == 1:
            count += 1
            continue
        # Check distinct vertices
        if len(set(m)) != k:
            return count
        for u, v in trees[i]:
            hu, hv = m[u], m[v]
            if hu == hv:
                return count
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                return count
            used_edges.add(e)
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit
    
    trees = instance_trees(20, 611)
    n = 20
    tree_infos = []
    for i, edges in enumerate(trees):
        k = i + 1
        tree_infos.append(_build_tree_info(edges, k))
    
    best_mappings = None
    best_score = 0
    
    # Try multiple greedy embeddings with different random seeds
    for trial_seed in range(20):
        if time.time() > deadline:
            break
        rng = Random(611 + trial_seed * 137)
        
        mappings, available, host_deg = _greedy_embed_all(trees, tree_infos, rng, deadline)
        
        if mappings is None:
            continue
        
        score = _count_embedded_edges(trees, mappings)
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m is not None else None for m in mappings]
        
        if best_score == 190:
            break
    
    # If we haven't found 190, try simulated annealing to improve
    if best_score < 190 and best_mappings is not None:
        if time.time() < deadline - 5:
            best_mappings = _anneal(trees, tree_infos, best_mappings, deadline)
            best_score = _count_embedded_edges(trees, best_mappings)
    
    # Build result
    result = [None] * 20
    if best_mappings is not None:
        for i in range(20):
            if best_mappings[i] is not None:
                result[i] = [int(x) for x in best_mappings[i]]
    
    # Verify
    if result:
        valid_count = _verify_packing(trees, result)
        # If verification fails, fall back to greedy-only result
        if valid_count < best_score:
            # Rebuild from verified
            result2 = [None] * 20
            for i in range(20):
                if result[i] is not None:
                    k = len(result[i])
                    if k <= 1:
                        result2[i] = result[i]
                    elif len(set(result[i])) == k:
                        # Check no self-loops in edges
                        ok = True
                        for u, v in trees[i]:
                            if result[i][u] == result[i][v]:
                                ok = False
                                break
                        if ok:
                            result2[i] = result[i]
            result = result2
    
    return result

def _anneal(trees, tree_infos, initial_mappings, deadline):
    """Simulated annealing on partial packings."""
    n = 20
    mappings = [m[:] if m is not None else None for m in initial_mappings]
    
    # Rebuild available edges and host degrees
    available = set((i, j) for i in range(n) for j in range(i + 1, n))
    host_deg = [0] * n
    for i, m in enumerate(mappings):
        if m is None or len(m) <= 1:
            continue
        for u, v in trees[i]:
            hu, hv = m[u], m[v]
            e = (min(hu, hv), max(hu, hv))
            available.discard(e)
            host_deg[hu] -= 1
            host_deg[hv] -= 1
    
    current_score = _count_embedded_edges(trees, mappings)
    best_mappings = [m[:] if m is not None else None for m in mappings]
    best_score = current_score
    
    rng = Random(42)
    T = 10.0  # Temperature
    
    while time.time() < deadline - 2:
        # Pick a random tree to move
        t_idx = rng.randint(0, 19)
        k = t_idx + 1
        
        if mappings[t_idx] is None:
            # Try to embed this tree
            if k == 1:
                h = rng.randint(0, 19)
                new_map = [h]
                # Check: no conflict (single vertex always ok)
                mappings[t_idx] = new_map
                new_score = current_score + 1
                if new_score > best_score:
                    best_score = new_score
                    best_mappings = [m[:] if m is not None else None for m in mappings]
                current_score = new_score
                # Update host_deg? For k=1, no edge changes
            else:
                edges, adj, deg, _ = tree_infos[t_idx]
                mapping = _embed_tree(edges, adj, deg, k, available, host_deg, rng, deadline, max_tries=20)
                if mapping is not None:
                    mappings[t_idx] = mapping
                    # Update available and host_deg
                    for u, v in edges:
                        hu, hv = mapping[u], mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        available.discard(e)
                        host_deg[hu] -= 1
                        host_deg[hv] -= 1
                    new_score = current_score + len(edges)
                    if new_score > best_score:
                        best_score = new_score
                        best_mappings = [m[:] if m is not None else None for m in mappings]
                    current_score = new_score
                else:
                    # Try to free up space by removing another tree temporarily
                    other_idx = rng.randint(0, 19)
                    if other_idx != t_idx and mappings[other_idx] is not None and len(mappings[other_idx]) > 1:
                        # Remove other tree
                        oe = trees[other_idx]
                        om = mappings[other_idx]
                        for u, v in oe:
                            hu, hv = om[u], om[v]
                            e = (min(hu, hv), max(hu, hv))
                            available.add(e)
                            host_deg[hu] += 1
                            host_deg[hv] += 1
                        mappings[other_idx] = None
                        freed = len(oe)
                        
                        # Try to embed t_idx
                        edges, adj, deg, _ = tree_infos[t_idx]
                        mapping = _embed_tree(edges, adj, deg, k, available, host_deg, rng, deadline, max_tries=30)
                        if mapping is not None:
                            mappings[t_idx] = mapping
                            for u, v in edges:
                                hu, hv = mapping[u], mapping[v]
                                e = (min(hu, hv), max(hu, hv))
                                available.discard(e)
                                host_deg[hu] -= 1
                                host_deg[hv] -= 1
                            # Try to re-embed other tree
                        oe2, oadj2, odeg2, ok2 = tree_infos[other_idx]
                        omap = _embed_tree(oe2, oadj2, odeg2, ok2, available, host_deg, rng, deadline, max_tries=30)
                        if omap is not None:
                            mappings[other_idx] = omap
                            for u, v in oe2:
                                hu, hv = omap[u], omap[v]
                                e = (min(hu, hv), max(hu, hv))
                                available.discard(e)
                                host_deg[hu] -= 1
                                host_deg[hv] -= 1
                            new_score = current_score + len(edges) + len(oe2)
                        else:
                            new_score = current_score + len(edges) - 0  # other tree still gone
                        if new_score > best_score:
                            best_score = new_score
                            best_mappings = [m[:] if m is not None else None for m in mappings]
                        current_score = new_score
                    continue
        else:
            # Move existing tree to new position
            edges, adj, deg, _ = tree_infos[t_idx]
            old_map = mappings[t_idx]
            
            # Remove old mapping
            for u, v in edges:
                hu, hv = old_map[u], old_map[v]
                e = (min(hu, hv), max(hu, hv))
                available.add(e)
                host_deg[hu] += 1
                host_deg[hv] += 1
            mappings[t_idx] = None
            
            # Try new embedding
            new_map = _embed_tree(edges, adj, deg, k, available, host_deg, rng, deadline, max_tries=30)
            if new_map is not None:
                mappings[t_idx] = new_map
                for u, v in edges:
                    hu, hv = new_map[u], new_map[v]
                    e = (min(hu, hv), max(hu, hv))
                    available.discard(e)
                    host_deg[hu] -= 1
                    host_deg[hv] -= 1
                # Score unchanged (same number of edges)
                if current_score > best_score:
                    best_score = current_score
                    best_mappings = [m[:] if m is not None else None for m in mappings]
            else:
                # Revert
                mappings[t_idx] = old_map
                for u, v in edges:
                    hu, hv = old_map[u], old_map[v]
                    e = (min(hu, hv), max(hu, hv))
                    available.discard(e)
                    host_deg[hu] -= 1
                    host_deg[hv] -= 1
        
        # Cool down
        T *= 0.995
    
    return best_mappings

if __name__ == "__main__":
    result = construct()
    trees = instance_trees(20, 611)
    total = 0
    for i, m in enumerate(result):
        if m is not None and len(m) > 1:
            total += len(trees[i])
    print(f"Total edges: {total}")
    print(f"Valid trees: {_verify_packing(trees, result)}")
