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
    # Safety margin: 145s to be safe under 150s limit
    deadline = start_time + 145.0
    
    trees = instance_trees(20, 611)
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # Sort trees by size descending (largest first) to constrain early
    # T_20 is most constrained in terms of mapping uniqueness if we consider edge usage?
    # Actually, larger trees use more edges, so they reduce the options for smaller trees more.
    order = list(range(19, -1, -1)) # 19, 18, ..., 0 (indices for T_20 ... T_1)
    
    # State
    used_edges = set()
    mappings = [None] * 20
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    def has_valid_host(tree_idx, v, mapping, used_hosts_in_tree):
        """Check if vertex v has at least one valid host."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        for h in range(20):
            if h in used_hosts_in_tree:
                continue
            valid = True
            for neighbor in adj_t[v]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
            if valid:
                return True
        return False

    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        if time_up():
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Find unmapped vertex with Minimum Remaining Values (MRV)
        best_vertex = -1
        min_options = 21
        best_options_list = []
        
        for i in range(k):
            if mapping[i] != -1:
                continue
            options = []
            for h in range(20):
                if h in used_hosts_in_tree:
                    continue
                valid = True
                for neighbor in adj_t[i]:
                    if mapping[neighbor] != -1:
                        hn = mapping[neighbor]
                        e = (min(h, hn), max(h, hn))
                        if e in used_edges:
                            valid = False
                            break
                if valid:
                    options.append(h)
            
            if len(options) < min_options:
                min_options = len(options)
                best_vertex = i
                best_options_list = options
                if min_options == 0:
                    return False
                if min_options == 1:
                    break
        
        if best_vertex == -1:
            # All mapped, commit
            for u in range(k):
                hu = mapping[u]
                for v in adj_t[u]:
                    if u < v:
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                        used_edges.add(e)
            mappings[tree_idx] = mapping[:]
            return True

        u = best_vertex
        
        # Heuristic: Try hosts in a deterministic order
        # Shuffling might help if the first choice leads to a dead end quickly,
        # but for this specific instance, deterministic often works.
        # We can try to prioritize hosts that have fewer available neighbors in the graph?
        # In K20, all hosts are symmetric initially.
        
        for h in best_options_list:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            
            # Forward Checking
            fc_failed = False
            for neighbor in adj_t[u]:
                if mapping[neighbor] == -1:
                    if not has_valid_host(tree_idx, neighbor, mapping, used_hosts_in_tree):
                        fc_failed = True
                        break
            
            if not fc_failed:
                if embed_tree(tree_idx, mapping, used_hosts_in_tree):
                    return True
            
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        if time_up():
            return False
            
        if idx_in_order == 20:
            return True
        
        t_idx = order[idx_in_order]
        k = t_idx + 1
        
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            return solve(idx_in_order + 1)
            
        return False

    # Run solver
    success = solve(0)
    
    # If success, mappings are filled.
    # If not, we might have partial mappings if we tracked best, but we return None for unembedded.
    # The task requires returning 20 entries.
    
    result = []
    for i in range(20):
        if mappings[i] is None:
            result.append(None)
        else:
            # Ensure Python ints
            result.append([int(x) for x in mappings[i]])
            
    return result
