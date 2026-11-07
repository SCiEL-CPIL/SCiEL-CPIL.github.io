import time
from random import Random
from itertools import combinations

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
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n = 20
    
    # Pre-process trees
    tree_infos = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_infos.append((edges, {}, [0], 1))
            continue
        
        adj = {v: [] for v in range(k)}
        degs = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degs[u] += 1
            degs[v] += 1
        
        tree_infos.append((edges, adj, degs, k))

    # Sort tree indices by complexity (size) descending
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-(len(trees[i])), -max(tree_infos[i][2]) if trees[i] else 0))

    # State: available edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    mappings = [None] * 20
    best_mappings = [None] * 20
    best_count = 0

    def compute_degrees(avail):
        degs = [0] * n
        for (a, b) in avail:
            degs[a] += 1
            degs[b] += 1
        return degs

    def backtrack(tree_pos, available_edges, host_degs):
        nonlocal best_mappings, best_count
        if time.time() > deadline:
            return False
            
        # Track progress for fallback
        if tree_pos > best_count:
            best_count = tree_pos
            for i in range(20):
                best_mappings[i] = mappings[i][:] if mappings[i] is not None else None

        if tree_pos == 20:
            return len(available_edges) == 0

        t_idx = tree_indices[tree_pos]
        edges, adj, degs, k = tree_infos[t_idx]
        
        if k == 1:
            mappings[t_idx] = [0]
            if backtrack(tree_pos + 1, available_edges, host_degs):
                return True
            mappings[t_idx] = None
            return False

        # Map highest degree vertices first
        tv_order = sorted(range(k), key=lambda v: -degs[v])
        
        host_map = {}
        
        def assign_tv(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all edges exist in available_edges
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                
                # Build new available set and update degrees
                used_edges = set()
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
                
                new_avail = available_edges - used_edges
                new_degs = list(host_degs)
                for (a, b) in used_edges:
                    new_degs[a] -= 1
                    new_degs[b] -= 1
                
                temp_map = [0] * k
                for tv, hv in host_map.items():
                    temp_map[tv] = hv
                mappings[t_idx] = temp_map
                
                if backtrack(tree_pos + 1, new_avail, new_degs):
                    return True
                
                mappings[t_idx] = None
                return False
            
            tv = tv_order[pos]
            deg_t = degs[tv]
            
            used_hosts = set(host_map.values())
            
            # Sort candidates by available degree descending for better pruning
            candidates = [h for h in range(n) if h not in used_hosts and host_degs[h] >= deg_t]
            candidates.sort(key=lambda h: -host_degs[h])
            
            for h in candidates:
                # Check neighbors already assigned
                valid = True
                for nb in adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                host_map[tv] = h
                if assign_tv(pos + 1):
                    return True
                del host_map[tv]
                
            return False

        if tree_pos == 0:
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if assign_tv(1):
                return True
            del host_map[tv0]
        else:
            if assign_tv(0):
                return True

        return False

    initial_degs = compute_degrees(all_edges)
    if backtrack(0, all_edges, initial_degs):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result
    else:
        # Return best partial solution found
        # Need to verify that the partial solution is actually valid (edge-disjoint)
        # The best_mappings were snapshotted at each level of backtrack
        # But we need to make sure they form a valid partial packing
        result = [None] * 20
        used_edges = set()
        valid = True
        for i in range(20):
            if best_mappings[i] is not None:
                mp = best_mappings[i]
                # Check distinctness
                if len(set(mp)) != len(mp):
                    valid = False
                    break
                # Check all vertices in range
                if any(x < 0 or x >= n for x in mp):
                    valid = False
                    break
                # Check edges
                edges_i = trees[i]
                for u, v in edges_i:
                    hu, hv = mp[u], mp[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        valid = False
                        break
                    used_edges.add(edge)
                if not valid:
                    break
        
        if valid:
            for i in range(20):
                if best_mappings[i] is not None:
                    result[i] = [int(x) for x in best_mappings[i]]
            return result
        else:
            return [None] * 20
