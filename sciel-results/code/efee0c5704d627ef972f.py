import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    trees = instance_trees(20, 611)
    
    # Build adjacency for each tree
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    total_tree_edges = sum(len(t) for t in trees)  # = 190
    
    # All edges in K20
    all_k20_edges = set()
    for i in range(20):
        for j in range(i+1, 20):
            all_k20_edges.add((i, j))
    
    best_result = [None] * 20
    best_edges = 0
    
    def try_greedy_packing(seed_val, order_mode):
        """Try a greedy randomized packing. Returns (result, embedded_edges)."""
        nonlocal deadline
        if time.time() > deadline:
            return None, 0
        
        rng = Random(seed_val)
        
        # Determine processing order
        if order_mode == 0:
            order = list(range(20, 0, -1))  # Largest first
        elif order_mode == 1:
            order = list(range(20, 0, -1))
            rng.shuffle(order)
        elif order_mode == 2:
            order = list(range(1, 21))  # Smallest first
        elif order_mode == 3:
            # Largest first but with some randomization among similar sizes
            order = list(range(20, 0, -1))
            for i in range(len(order)-1):
                if rng.random() < 0.3:
                    order[i], order[i+1] = order[i+1], order[i]
        else:
            # Degree-based: trees with higher max degree first
            order = list(range(20, 0, -1))
            max_degrees = []
            for k in range(1, 21):
                if k <= 2:
                    max_degrees.append(0 if k == 1 else 1)
                else:
                    adj = tree_adj[k-1]
                    max_degrees.append(max(len(adj[i]) for i in range(k)))
            # Sort by max degree descending, then by size descending
            indexed = list(zip(range(1, 21), max_degrees))
            indexed.sort(key=lambda x: (-x[1], -x[0]))
            order = [x[0] for x in indexed]
        
        used_edges = set()
        result = [None] * 20
        embedded_edges = 0
        
        # Track residual degrees of host vertices
        residual_deg = [19] * 20  # Each vertex in K20 has degree 19 initially
        
        for k in order:
            if time.time() > deadline:
                return result, embedded_edges
            
            if k == 1:
                # Pick a vertex with max residual degree
                best_h = max(range(20), key=lambda h: residual_deg[h])
                result[0] = [best_h]
                embedded_edges += 0
                continue
            
            if k == 2:
                # Pick an unused edge; prefer edges between high-residual vertices
                available = [(i, j) for (i, j) in all_k20_edges if (i, j) not in used_edges]
                if not available:
                    break
                # Score by sum of residual degrees
                available.sort(key=lambda e: -(residual_deg[e[0]] + residual_deg[e[1]]))
                # Take from top with some randomness
                top_n = min(5, len(available))
                edge = rng.choice(available[:top_n])
                used_edges.add(edge)
                result[1] = [edge[0], edge[1]]
                residual_deg[edge[0]] -= 1
                residual_deg[edge[1]] -= 1
                embedded_edges += 1
                continue
            
            # For k >= 3: greedy embedding with backtracking fallback
            adj = tree_adj[k-1]
            k_val = k
            num_edges = k_val - 1
            
            # Order tree vertices: high degree first (more constrained)
            deg = [len(adj[i]) for i in range(k_val)]
            v_order = sorted(range(k_val), key=lambda x: (-deg[x], x))
            
            mapping = [-1] * k_val
            host_used = [False] * 20
            success = False
            
            # Try multiple random attempts for this tree
            max_attempts = 20
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                mapping = [-1] * k_val
                host_used = [False] * 20
                
                # Greedy assignment in v_order
                feasible = True
                for idx in range(k_val):
                    tv = v_order[idx]
                    assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
                    
                    candidates = []
                    for h in range(20):
                        if host_used[h]:
                            continue
                        ok = True
                        for nb in assigned_nbs:
                            hnb = mapping[nb]
                            edge = (min(h, hnb), max(h, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                        if ok:
                            candidates.append(h)
                    
                    if not candidates:
                        feasible = False
                        break
                    
                    # Score candidates: prefer those that maintain high residual degrees
                    # and don't create bottlenecks
                    def score_h(h):
                        # Primary: residual degree (higher is better)
                        score = residual_deg[h] * 10
                        # Secondary: count of unused edges to unassigned tree vertices
                        # (more flexibility)
                        unassigned_tree_verts = [tv2 for tv2 in range(k_val) if mapping[tv2] == -1 and tv2 != tv]
                        flex = 0
                        for tv2 in unassigned_tree_verts:
                            if tv2 in adj[tv]:  # Only count if adjacent in tree
                                for h2 in range(20):
                                    if not host_used[h2] and h2 != h:
                                        e = (min(h, h2), max(h, h2))
                                        if e not in used_edges:
                                            flex += 1
                        score += flex
                        # Tertiary: random tiebreaker
                        score += rng.random()
                        return score
                    
                    candidates.sort(key=score_h, reverse=True)
                    
                    # Take from top with some randomness
                    top_n = min(3, len(candidates))
                    h = rng.choice(candidates[:top_n])
                    mapping[tv] = h
                    host_used[h] = True
                
                if feasible:
                    # Verify all edges are valid
                    valid = True
                    for (u, v) in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        edge = (min(hu, hv), max(hu, hv))
                        if edge in used_edges:
                            valid = False
                            break
                    if valid:
                        success = True
                        break
            
            if not success:
                # Try a small backtracking search as fallback
                tree_deadline = time.time() + 2
                node_count = [0]
                max_nodes = 100000
                
                def backtrack(idx):
                    if time.time() > tree_deadline or node_count[0] > max_nodes:
                        return False
                    node_count[0] += 1
                    if idx == k_val:
                        return True
                    
                    tv = v_order[idx]
                    assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
                    
                    candidates = []
                    for h in range(20):
                        if host_used[h]:
                            continue
                        ok = True
                        for nb in assigned_nbs:
                            hnb = mapping[nb]
                            edge = (min(h, hnb), max(h, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                        if ok:
                            candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    # Sort by residual degree (descending)
                    candidates.sort(key=lambda h: -residual_deg[h])
                    
                    for h in candidates:
                        mapping[tv] = h
                        host_used[h] = True
                        
                        if backtrack(idx + 1):
                            return True
                        
                        mapping[tv] = -1
                        host_used[h] = False
                    
                    return False
                
                # Reset for backtracking
                mapping = [-1] * k_val
                host_used = [False] * 20
                if backtrack(0):
                    success = True
            
            if not success:
                return result, embedded_edges
            
            # Commit the mapping
            result[k-1] = [int(mapping[i]) for i in range(k_val)]
            for (u, v) in trees[k-1]:
                hu, hv = mapping[u], mapping[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
                residual_deg[hu] -= 1
                residual_deg[hv] -= 1
            embedded_edges += num_edges
        
        return result, embedded_edges
    
    # Try multiple configurations
    num_attempts = 0
    for order_mode in range(5):
        for seed_offset in range(100):
            if time.time() > deadline:
                break
            num_attempts += 1
            seed_val = num_attempts * 31337 + 42
            result, edges = try_greedy_packing(seed_val, order_mode)
            if result is not None and edges > best_edges:
                best_edges = edges
                best_result = result[:]
                if best_edges == total_tree_edges:
                    return best_result
    
    # Final attempt: if we have a near-complete solution, try to repair
    if best_edges >= 180 and best_edges < total_tree_edges:
        # Identify which trees are missing
        missing_trees = [k for k in range(1, 21) if best_result[k-1] is None]
        if missing_trees:
            # Reconstruct used edges
            used_edges = set()
            for k in range(1, 21):
                if best_result[k-1] is not None and k >= 2:
                    mapping = best_result[k-1]
                    for (u, v) in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
            
            residual_deg = [19] * 20
            for e in used_edges:
                residual_deg[e[0]] -= 1
                residual_deg[e[1]] -= 1
            
            for k in missing_trees:
                if time.time() > deadline:
                    break
                if k == 1:
                    best_h = max(range(20), key=lambda h: residual_deg[h])
                    best_result[0] = [best_h]
                    best_edges += 0
                    continue
                if k == 2:
                    available = [(i, j) for (i, j) in all_k20_edges if (i, j) not in used_edges]
                    if available:
                        available.sort(key=lambda e: -(residual_deg[e[0]] + residual_deg[e[1]]))
                        edge = available[0]
                        used_edges.add(edge)
                        best_result[1] = [edge[0], edge[1]]
                        residual_deg[edge[0]] -= 1
                        residual_deg[edge[1]] -= 1
                        best_edges += 1
                    continue
                
                adj = tree_adj[k-1]
                k_val = k
                deg = [len(adj[i]) for i in range(k_val)]
                v_order = sorted(range(k_val), key=lambda x: (-deg[x], x))
                
                mapping = [-1] * k_val
                host_used = [False] * 20
                # Mark vertices already used by other trees
                for kk in range(1, 21):
                    if kk != k and best_result[kk-1] is not None:
                        for h in best_result[kk-1]:
                            host_used[h] = True
                
                tree_deadline = time.time() + 5
                node_count = [0]
                max_nodes = 200000
                
                def backtrack(idx):
                    if time.time() > tree_deadline or node_count[0] > max_nodes:
                        return False
                    node_count[0] += 1
                    if idx == k_val:
                        return True
                    
                    tv = v_order[idx]
                    assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
                    
                    candidates = []
                    for h in range(20):
                        if host_used[h]:
                            continue
                        ok = True
                        for nb in assigned_nbs:
                            hnb = mapping[nb]
                            edge = (min(h, hnb), max(h, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                        if ok:
                            candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    candidates.sort(key=lambda h: -residual_deg[h])
                    
                    for h in candidates:
                        mapping[tv] = h
                        host_used[h] = True
                        
                        if backtrack(idx + 1):
                            return True
                        
                        mapping[tv] = -1
                        host_used[h] = False
                    
                    return False
                
                if backtrack(0):
                    best_result[k-1] = [int(mapping[i]) for i in range(k_val)]
                    for (u, v) in trees[k-1]:
                        hu, hv = mapping[u], mapping[v]
                        edge = (min(hu, hv), max(hu, hv))
                        used_edges.add(edge)
                        residual_deg[hu] -= 1
                        residual_deg[hv] -= 1
                    best_edges += (k_val - 1)
    
    return best_result
