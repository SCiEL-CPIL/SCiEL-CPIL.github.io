import time
import random
import sys

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_hosts = 20
    
    # Pre-compute tree properties
    tree_edges = []
    tree_adj = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_edges.append(edges)
        tree_adj.append(adj)
        
    # Sort trees by size descending (largest first) for better pruning
    order = list(range(n_trees))
    order.sort(key=lambda x: -(x + 1))
    
    # Map original index to sorted index
    # We will store results in original index order
    
    # Bitmask for used edges in K20
    # K20 has 20*19/2 = 190 edges.
    # We can represent the set of used edges as a set of integers (edge ids).
    # Edge id for (u, v) with u < v: u * 20 + v (simple and unique enough for set)
    
    used_edges = set()
    mappings = [None] * n_trees
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def check_time():
        return time.time() > deadline

    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        if check_time():
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with highest constraint
        # Constraint: number of mapped neighbors
        best_vertex = -1
        max_mapped_neighbors = -1
        max_degree = -1
        
        for i in range(k):
            if mapping[i] != -1:
                continue
            mapped_neighbors = 0
            for neighbor in adj_t[i]:
                if mapping[neighbor] != -1:
                    mapped_neighbors += 1
            deg_i = len(adj_t[i])
            if best_vertex == -1:
                best_vertex = i
                max_mapped_neighbors = mapped_neighbors
                max_degree = deg_i
            elif mapped_neighbors > max_mapped_neighbors:
                best_vertex = i
                max_mapped_neighbors = mapped_neighbors
                max_degree = deg_i
            elif mapped_neighbors == max_mapped_neighbors and deg_i > max_degree:
                best_vertex = i
                max_degree = deg_i
                
        if best_vertex == -1:
            # All mapped
            # Verify all edges are valid (not in used_edges)
            for u, v in tree_edges[tree_idx]:
                hu = mapping[u]
                hv = mapping[v]
                if hu > hv:
                    hu, hv = hv, hu
                if (hu, hv) in used_edges:
                    return False
            # Commit
            for u, v in tree_edges[tree_idx]:
                hu = mapping[u]
                hv = mapping[v]
                if hu > hv:
                    hu, hv = hv, hu
                used_edges.add((hu, hv))
            mappings[tree_idx] = mapping[:]
            return True

        u = best_vertex
        valid_candidates = []
        for h in range(n_hosts):
            if h in used_hosts_in_tree:
                continue
            valid = True
            for neighbor in adj_t[u]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    if h > hn:
                        h, hn = hn, h
                    if (h, hn) in used_edges:
                        valid = False
                        break
            if valid:
                valid_candidates.append(h)
                
        if not valid_candidates:
            return False

        # LCV: Sort candidates by how many options they leave for neighbors
        # Approximate LCV: Count valid hosts for each unmapped neighbor
        if len(valid_candidates) > 1:
            def count_options(h):
                total = 0
                for neighbor in adj_t[u]:
                    if mapping[neighbor] == -1:
                        valid_count = 0
                        for hn in range(n_hosts):
                            if hn in used_hosts_in_tree:
                                continue
                            # Check if hn is valid for neighbor given u->h
                            # Neighbor's neighbors that are mapped must not conflict with hn
                            ok = True
                            for n_neighbor in adj_t[neighbor]:
                                if mapping[n_neighbor] != -1:
                                    hnn = mapping[n_neighbor]
                                    if hn > hnn:
                                        hn, hnn = hnn, hn
                                    if (hn, hnn) in used_edges:
                                        ok = False
                                        break
                            if ok:
                                valid_count += 1
                        total += valid_count
                return total
            
            valid_candidates.sort(key=lambda h: -count_options(h))
            # Add slight randomness to break ties and diversify search
            if len(valid_candidates) > 3:
                top_k = min(3, len(valid_candidates))
                valid_candidates[:top_k] = [valid_candidates[i] for i in random.sample(range(top_k), top_k)]
        else:
            pass # Only one candidate

        for h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            
            # Forward Checking
            fc_failed = False
            for neighbor in adj_t[u]:
                if mapping[neighbor] == -1:
                    # Check if neighbor has any valid host
                    has_valid = False
                    for hn in range(n_hosts):
                        if hn in used_hosts_in_tree:
                            continue
                        # Check if hn is valid for neighbor
                        ok = True
                        for n_neighbor in adj_t[neighbor]:
                            if mapping[n_neighbor] != -1:
                                hnn = mapping[n_neighbor]
                                if hn > hnn:
                                    hn, hnn = hnn, hn
                                if (hn, hnn) in used_edges:
                                    ok = False
                                    break
                        if ok:
                            has_valid = True
                            break
                    if not has_valid:
                        fc_failed = True
                        break
            
            if not fc_failed:
                if embed_tree(tree_idx, mapping, used_hosts_in_tree):
                    return True
            
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        nonlocal best_score, best_mappings
        if idx_in_order == n_trees:
            # All trees embedded
            score = 190
            if score > best_score:
                best_score = score
                for i in range(n_trees):
                    best_mappings[i] = mappings[i][:] if mappings[i] else None
            return True
        
        if check_time():
            return False
            
        t_idx = order[idx_in_order]
        k = t_idx + 1
        
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            return solve(idx_in_order + 1)
            
        return False

    # Run solver with multiple restarts if needed
    # The first run is deterministic (seeded RNG for tie-breaking)
    # If it fails, we can try again with different seeds
    
    attempts = 0
    while time.time() < deadline:
        # Reset state
        used_edges = set()
        mappings = [None] * n_trees
        random.seed(12345 + attempts)
        
        success = solve(0)
        attempts += 1
        
        if success:
            break
            
        # Calculate score of current partial packing (if any)
        # In this implementation, mappings are only set on success of a tree,
        # but if a tree fails, its mapping is not committed.
        # However, previous trees are committed.
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(trees[i])
                
        if score > best_score:
            best_score = score
            for i in range(n_trees):
                best_mappings[i] = mappings[i][:] if mappings[i] else None
                
        if best_score == 190:
            break

    # Ensure return type is correct
    result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    return result
