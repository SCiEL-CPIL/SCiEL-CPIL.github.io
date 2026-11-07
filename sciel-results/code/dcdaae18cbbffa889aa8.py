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
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit, leave 5s buffer)
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Pre-process trees
    # tree_data[i] = (edges, adj, degs, k)
    tree_data = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_data.append((edges, {0: []}, [0], k))
            continue
        
        adj = {v: [] for v in range(k)}
        degs = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degs[u] += 1
            degs[v] += 1
        tree_data.append((edges, adj, degs, k))

    # Order trees: Largest first (most edges), then highest max degree
    # This helps in pruning early
    tree_order = list(range(20))
    tree_order.sort(key=lambda i: (-(len(trees[i])), -max(tree_data[i][2]) if trees[i] else 0))
    
    # Initial available edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    # Initial degrees of hosts in K20 (all 19)
    host_degs = [19] * n
    
    mappings = [None] * 20
    best_mappings = [None] * 20
    best_valid_count = 0
    
    def update_best():
        nonlocal best_mappings, best_valid_count
        count = sum(1 for m in mappings if m is not None)
        if count > best_valid_count:
            best_valid_count = count
            for i in range(20):
                if mappings[i] is not None:
                    best_mappings[i] = mappings[i][:]
                else:
                    best_mappings[i] = None

    def is_feasible(pos, avail_edges, h_degs):
        """Check if remaining trees can still be packed given current state."""
        # For each remaining tree, check if there are enough available edges
        # and if host degrees are sufficient
        remaining_trees = [tree_order[p] for p in range(pos, 20)]
        
        for t_idx in remaining_trees:
            edges, adj, t_degs, k = tree_data[t_idx]
            if k == 1:
                continue
            # Need k-1 edges for this tree
            if len(avail_edges) < k - 1:
                return False
            # Check if max degree in tree can be satisfied by any host
            max_deg = max(t_degs)
            if max(h_degs) < max_deg:
                return False
            # More detailed: sum of degrees in tree = 2*(k-1)
            # Sum of host degrees available must be >= 2*(k-1) for this tree
            # But this is a weak constraint; skip for speed
        
        return True

    def backtrack(pos, avail_edges, h_degs):
        nonlocal best_mappings, best_valid_count, mappings
        
        if time.time() > deadline:
            return False
            
        if pos == 20:
            return len(avail_edges) == 0
        
        # Forward checking: verify feasibility before proceeding
        if not is_feasible(pos, avail_edges, h_degs):
            return False
            
        t_idx = tree_order[pos]
        edges, adj, t_degs, k = tree_data[t_idx]
        
        if k == 1:
            mappings[t_idx] = [0]
            update_best()
            if backtrack(pos + 1, avail_edges, h_degs):
                return True
            mappings[t_idx] = None
            return False

        tv_order = sorted(range(k), key=lambda v: -t_degs[v])
        host_map = {}
        
        def assign_tv(t_pos):
            if time.time() > deadline:
                return False
                
            if t_pos == k:
                used_edges = set()
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    used_edges.add(edge)
                
                new_avail = avail_edges - used_edges
                new_degs = list(h_degs)
                for a, b in used_edges:
                    new_degs[a] -= 1
                    new_degs[b] -= 1
                
                temp_map = [0] * k
                for tv, hv in host_map.items():
                    temp_map[tv] = hv
                mappings[t_idx] = temp_map
                update_best()
                
                if backtrack(pos + 1, new_avail, new_degs):
                    return True
                
                mappings[t_idx] = None
                return False
            
            tv = tv_order[t_pos]
            deg_t = t_degs[tv]
            used_hosts = set(host_map.values())
            
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                if h_degs[h] < deg_t:
                    continue
                
                valid = True
                for nb in adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in avail_edges:
                            valid = False
                            break
                if valid:
                    candidates.append((h_degs[h], h))
                    
            candidates.sort()
            
            for _, h in candidates:
                host_map[tv] = h
                if assign_tv(t_pos + 1):
                    return True
                del host_map[tv]
                
            return False

        if pos == 0:
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if assign_tv(1):
                return True
            del host_map[tv0]
        else:
            if assign_tv(0):
                return True
                
        return False

    if backtrack(0, all_edges, host_degs):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result
    else:
        # Return best valid partial packing
        result = [None] * 20
        for i in range(20):
            if best_mappings[i] is not None:
                result[i] = [int(x) for x in best_mappings[i]]
        return result
