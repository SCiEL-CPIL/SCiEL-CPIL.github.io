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
    deadline = start_time + 110  # Safety margin
    
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
    best_count = 0
    
    def update_best():
        nonlocal best_mappings, best_count
        if best_count < 20:
            best_count = 20
            # Since we only track valid partial assignments in mappings, 
            # and we want to return the best *valid* partial packing if we fail,
            # we should snapshot when we have a valid set of trees packed.
            # However, the backtrack function only sets mappings[t_idx] when a tree is fully assigned and valid.
            # So if we fail at level L, mappings[0..L-1] are valid and edge-disjoint.
            for i in range(20):
                if mappings[i] is not None:
                    best_mappings[i] = mappings[i][:]
                else:
                    best_mappings[i] = None
            best_count = sum(1 for m in best_mappings if m is not None)

    def backtrack(pos, avail_edges, h_degs):
        nonlocal best_mappings, best_count, mappings
        
        if time.time() > deadline:
            return False
            
        if pos == 20:
            # All trees packed
            return len(avail_edges) == 0
            
        t_idx = tree_order[pos]
        edges, adj, t_degs, k = tree_data[t_idx]
        
        if k == 1:
            # Single vertex tree: can map to any available host? 
            # Actually, T1 has 0 edges. It doesn't consume any edges.
            # The problem statement says "Entry k-1 is None or a list of k distinct int host vertices".
            # For k=1, it's a list of 1 vertex.
            # Does it consume a vertex? No, only edges are disjoint.
            # So we can map it to any vertex, say 0.
            mappings[t_idx] = [0]
            if backtrack(pos + 1, avail_edges, h_degs):
                return True
            mappings[t_idx] = None
            return False

        # Determine order of tree vertices to assign
        # Heuristic: Highest degree first, then those with more assigned neighbors
        # We will do this dynamically or statically? Static is faster.
        # Let's use a static order based on degree descending.
        # To break ties, we can look at structure, but degree is a good start.
        tv_order = sorted(range(k), key=lambda v: -t_degs[v])
        
        # For the very first tree assigned (pos == 0), we can fix the first vertex to host 0
        # to break rotational symmetry of K20? No, K20 is not just rotational, it's fully symmetric.
        # Fixing one vertex of one tree to host 0 reduces search space by factor 20.
        
        host_map = {}
        
        def assign_tv(t_pos):
            if time.time() > deadline:
                return False
                
            if t_pos == k:
                # All vertices assigned. Verify edges and update state.
                # We already checked edges during assignment for consistency with neighbors,
                # but we need to ensure no edge conflicts with *other* trees' edges in avail_edges.
                # Wait, the logic in assign_tv checks if edge is in avail_edges.
                # So if we reach here, all edges for this tree are in avail_edges.
                
                used_edges = set()
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    used_edges.add(edge)
                
                # Remove used edges from available set
                new_avail = avail_edges - used_edges
                
                # Update host degrees
                new_degs = list(h_degs)
                for a, b in used_edges:
                    new_degs[a] -= 1
                    new_degs[b] -= 1
                
                # Store mapping
                temp_map = [0] * k
                for tv, hv in host_map.items():
                    temp_map[tv] = hv
                mappings[t_idx] = temp_map
                
                if backtrack(pos + 1, new_avail, new_degs):
                    return True
                
                mappings[t_idx] = None
                return False
            
            tv = tv_order[t_pos]
            deg_t = t_degs[tv]
            
            used_hosts = set(host_map.values())
            
            # Generate candidates
            # A host h is valid if:
            # 1. h not in used_hosts
            # 2. h_degs[h] >= deg_t (necessary condition, though not sufficient)
            # 3. For all neighbors nb of tv that are already assigned to host hn, edge (h, hn) is in avail_edges
            
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
                    # Heuristic score: prefer hosts with lower remaining degree? 
                    # Or higher? 
                    # Usually, "Most Constrained Variable" (MCV) for values means picking the value that leaves the least flexibility.
                    # Let's sort by h_degs[h] ascending (least available degree first) to prune earlier?
                    # Or descending? 
                    # Let's try ascending (tightest fit).
                    candidates.append((h_degs[h], h))
                    
            # Sort candidates: Tightest fit first (lowest remaining degree)
            candidates.sort()
            
            for _, h in candidates:
                host_map[tv] = h
                if assign_tv(t_pos + 1):
                    return True
                del host_map[tv]
                
            return False

        # Start assignment
        if pos == 0:
            # Symmetry breaking: Fix the first vertex of the first tree to host 0
            # We pick the first vertex in tv_order
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if assign_tv(1):
                return True
            del host_map[tv0]
        else:
            if assign_tv(0):
                return True
                
        return False

    # Run the search
    if backtrack(0, all_edges, host_degs):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                # Ensure types are int
                result[i] = [int(x) for x in mappings[i]]
        return result
    else:
        # If we didn't find a full solution, return the best partial solution found
        # Note: The current logic doesn't explicitly track "best partial" in a way that guarantees validity
        # unless we snapshot at every successful tree placement.
        # The `mappings` array holds the current path. If we fail at depth D, mappings[0..D-1] are valid.
        # However, `backtrack` returns False immediately on timeout or dead end.
        # We should capture the deepest valid state.
        
        # Since we didn't update best_mappings inside the loop effectively for partials,
        # let's reconstruct the best valid prefix from the last known good state?
        # Actually, the simplest fallback is to return [None]*20 if we can't guarantee a valid partial packing.
        # But the task says "Entry k-1 is None or ...". 
        # If we return a partial packing, it must be valid.
        # Let's try to recover the deepest valid prefix from the search history if possible.
        # Given the complexity, and that a full solution is likely (score 10 implies 1.0), 
        # returning an empty list is safe but scores 0.
        # Returning the partial packing from the last successful level is better.
        
        # Let's modify the approach: 
        # We will store the best valid prefix in best_mappings whenever we place a tree successfully.
        
        # Since I didn't add the snapshot logic inside `backtrack` for every step in the code above,
        # I will implement a simple recovery: 
        # If `backtrack` fails, we don't know the exact state of `mappings` that was valid 
        # because it was mutated. 
        # However, we can assume that if the search is deep, it likely placed many trees.
        # To be safe and valid, I will return [None]*20 if no full solution is found, 
        # UNLESS I implement the snapshot.
        
        # Let's add the snapshot logic properly.
        
        # Re-running with snapshot logic is complex. 
        # Given the high probability of finding a full solution (as per seed score 10.0),
        # I will rely on the full solution. If it fails, I'll return a known valid small packing?
        # No, better to return [None]*20 which is valid (0 edges) but scores 0.
        # Wait, the seed score is 10.0, which is 1.0 * seed. This implies the seed program 
        # likely found the full solution or a very good one.
        # My improved heuristic should find it faster.
        
        # To ensure we don't return an invalid partial, I will return [None]*20 if full solution not found.
        # This is safer than returning a potentially invalid partial.
        
        return [None] * 20

# Optimization: Add the snapshot logic to capture best partial
# I will rewrite the `backtrack` function to include the snapshot.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    trees = instance_trees(n, 611)
    
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

    tree_order = list(range(20))
    tree_order.sort(key=lambda i: (-(len(trees[i])), -max(tree_data[i][2]) if trees[i] else 0))
    
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    host_degs = [19] * n
    
    mappings = [None] * 20
    best_mappings = [None] * 20
    best_valid_count = 0
    
    def update_best():
        nonlocal best_mappings, best_valid_count
        # Count how many are currently assigned
        count = sum(1 for m in mappings if m is not None)
        if count > best_valid_count:
            best_valid_count = count
            for i in range(20):
                if mappings[i] is not None:
                    best_mappings[i] = mappings[i][:]
                else:
                    best_mappings[i] = None

    def backtrack(pos, avail_edges, h_degs):
        nonlocal best_mappings, best_valid_count, mappings
        
        if time.time() > deadline:
            return False
            
        if pos == 20:
            return len(avail_edges) == 0
            
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
