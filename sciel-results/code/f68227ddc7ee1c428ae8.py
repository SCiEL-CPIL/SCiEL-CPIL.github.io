"""Packing trees T1..T20 into K20 via backtracking with constraint propagation."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    n_trees = len(trees)
    
    # Sort trees by size (number of edges) descending for better pruning
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    # Precompute for each tree: adjacency and a good vertex ordering
    tree_info = []
    for idx in tree_indices:
        tree = trees[idx]
        k = len(tree) + 1 if tree else 1
        if k == 1:
            tree_info.append(None)
            continue
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build ordering: start from a leaf, then greedily pick the most
        # constrained vertex (most neighbors already in order)
        if k <= 2:
            order = list(range(k))
        else:
            # Find a leaf to start
            leaves = [v for v in range(k) if len(adj[v]) == 1]
            start = leaves[0] if leaves else 0
            order = [start]
            remaining = set(range(k)) - {start}
            order_set = {start}
            
            while remaining:
                best_v = None
                best_score = -1
                for v in remaining:
                    # Score: number of neighbors already in order
                    score = sum(1 for nb in adj[v] if nb in order_set)
                    # Tie-break: higher degree first
                    if score > best_score:
                        best_score = score
                        best_v = v
                    elif score == best_score and best_v is not None:
                        if len(adj[v]) > len(adj[best_v]):
                            best_v = v
                order.append(best_v)
                remaining.discard(best_v)
                order_set.add(best_v)
        
        order_index = {tv: i for i, tv in enumerate(order)}
        tree_info.append((adj, order, order_index))
    
    used_edges = set()
    result = [None] * n_trees
    tree_info_lookup = {tree_indices[i]: tree_info[i] for i in range(n_trees)}
    
    def find_embedding(tree_idx, deadline, rng):
        """Find an embedding of the given tree using unused edges."""
        info = tree_info_lookup[tree_idx]
        if info is None:
            return [0]
        adj, order, order_index = info
        k = len(order)
        host_assignment = [None] * k
        available = set(range(20))
        
        # Precompute: for each position in order, which neighbors are "before" in order
        prev_neighbors = [[] for _ in range(k)]
        next_neighbors = [[] for _ in range(k)]
        for i, tv in enumerate(order):
            for nb in adj[tv]:
                nb_idx = order_index[nb]
                if nb_idx < i:
                    prev_neighbors[i].append(nb_idx)
                elif nb_idx > i:
                    next_neighbors[i].append(nb_idx)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            tv = order[idx]
            # Get available vertices
            avail_list = list(available)
            
            # Value ordering: prefer vertices that preserve residual degree for future trees
            def value_score(hv):
                # Count how many unused edges hv has to available vertices
                # (higher is better - preserves flexibility)
                residual = 0
                for nh in available:
                    if nh == hv:
                        continue
                    if edge_key(hv, nh) not in used_edges:
                        residual += 1
                # Also penalize if hv connects to already-assigned neighbors via used edges
                penalty = 0
                for nb_idx in prev_neighbors[idx]:
                    hnb = host_assignment[nb_idx]
                    if edge_key(hv, hnb) in used_edges:
                        penalty += 100
                return (penalty, -residual)
            
            avail_list.sort(key=value_score)
            
            if rng is not None:
                # Add slight randomness to break ties
                for i in range(len(avail_list) - 1, 0, -1):
                    if rng.random() < 0.15:
                        j = rng.randint(0, i)
                        avail_list[i], avail_list[j] = avail_list[j], avail_list[i]
            
            for hv in avail_list:
                # Check conflicts with already-assigned neighbors
                conflict = False
                for nb_idx in prev_neighbors[idx]:
                    hnb = host_assignment[nb_idx]
                    ek = edge_key(hv, hnb)
                    if ek in used_edges:
                        conflict = True
                        break
                if conflict:
                    continue
                
                host_assignment[idx] = hv
                available.remove(hv)
                
                # Lightweight forward checking: just verify each unassigned neighbor
                # has at least one available vertex it can connect to
                fc_ok = True
                for nb_idx in next_neighbors[idx]:
                    found = False
                    for nh in available:
                        if edge_key(hv, nh) in used_edges:
                            continue
                        found = True
                        break
                    if not found:
                        fc_ok = False
                        break
                
                if fc_ok and backtrack(idx + 1):
                    return True
                
                available.add(hv)
                host_assignment[idx] = None
            
            return False
        
        if backtrack(0):
            mapping = [None] * k
            for i, tv in enumerate(order):
                mapping[tv] = host_assignment[i]
            for u, v in trees[tree_idx]:
                used_edges.add(edge_key(mapping[u], mapping[v]))
            return mapping
        
        return None
    
    # Process trees from largest to smallest
    for ti, tree_idx in enumerate(tree_indices):
        if time.time() > deadline:
            break
        
        tree = trees[tree_idx]
        k = len(tree) + 1 if tree else 1
        
        if k == 1:
            result[tree_idx] = [0]
            continue
        
        # Try with deterministic order first, then with random shuffling if needed
        mapping = find_embedding(tree_idx, deadline, None)
        if mapping is None:
            # Try with randomized order - multiple attempts with different seeds
            base_seed = tree_idx * 7919 + 13
            for attempt in range(20):
                if time.time() > deadline:
                    break
                rng = Random(base_seed + attempt * 104729)
                mapping = find_embedding(tree_idx, deadline, rng)
                if mapping is not None:
                    break
        if mapping is None:
            break
        result[tree_idx] = mapping
    
    # Verify and return
    if all(r is not None for r in result):
        all_used = set()
        for i in range(n_trees):
            tree = trees[i]
            mapping = result[i]
            for u, v in tree:
                ek = edge_key(mapping[u], mapping[v])
                if ek in all_used:
                    return result  # Should not happen, but safe
                all_used.add(ek)
        return result
    
    return result
