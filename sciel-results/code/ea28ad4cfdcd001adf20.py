import time
from random import Random
from itertools import permutations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            if k == 1:
                trees.append([])
            else:
                trees.append([(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            # Find the smallest leaf
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
    deadline = start_time + 110.0 # Safety margin
    
    trees = instance_trees(20, 611)
    
    # Sort trees by size descending to embed large ones first
    # We need to keep track of original indices to return in correct order
    # Tree k has k-1 edges.
    indexed_trees = [(len(t), i, t) for i, t in enumerate(trees)]
    indexed_trees.sort(key=lambda x: x[0], reverse=True)
    
    # Total edges in K20
    all_edges = set()
    for u in range(20):
        for v in range(u+1, 20):
            all_edges.add((u, v))
    
    def try_pack(seed_offset=0):
        remaining_edges = set(all_edges)
        results = [None] * 20
        rng = Random(12345 + seed_offset)
        
        # We process trees in the sorted order (largest first)
        # But we might want to randomize the order of attempts for same size or just strict size.
        # Strict size is usually better for constraints.
        
        # Backtracking state
        # We will recursively try to embed indexed_trees[0], then [1], etc.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(indexed_trees):
                return True
            
            _, orig_idx, tree_edges = indexed_trees[idx]
            k = len(tree_edges) + 1 if tree_edges else 1
            
            # Find an embedding of this tree into remaining_edges
            # If k == 1, any vertex works.
            # For k > 1, we use DFS on the tree structure.
            
            if k == 1:
                # Tree is a single vertex. 
                # The mapping is just [host_vertex].
                # Since it has no edges, it doesn't consume any edges from remaining_edges.
                # We can map it to any vertex. Let's pick one randomly or deterministically.
                # To be safe and valid, we just need to return a list of 1 int.
                # The problem says "Entry k-1 is None or a list of k distinct int host vertices".
                # For T1, it's a list of 1 vertex.
                # We can pick vertex 0.
                results[orig_idx] = [0]
                if backtrack(idx + 1):
                    return True
                # If backtrack fails, we might need to try a different vertex?
                # No, T1 has no edges, so its vertex choice doesn't affect edge availability.
                # So if it fails later, T1's choice wasn't the cause.
                # However, distinctness is required for vertices in the *same* tree.
                # T1 has only 1 vertex, so distinctness is trivial.
                # So if it fails, it's not because of T1.
                # But wait, the problem says "list of k distinct int host vertices".
                # It does NOT say host vertices must be distinct across different trees.
                # Only edges must be disjoint.
                # So T1 can always be mapped to 0.
                return False

            # Build adjacency for the tree
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Root the tree at a high-degree node to reduce branching
            root = max(range(k), key=lambda x: len(adj[x]))
            
            # We need to find a mapping of tree vertices 0..k-1 to host vertices 0..19
            # such that for every edge (u,v) in tree, (map[u], map[v]) is in remaining_edges.
            # map[u] must be distinct from map[v] for u!=v (implied by edges being distinct in K20, 
            # but also tree vertices are distinct, so their images must be distinct? 
            # "list of k distinct int host vertices" -> YES, images must be distinct).
            
            # We use DFS to assign host vertices to tree vertices.
            # We fix the root's image to break symmetry? 
            # No, we must try all possible host vertices for the root.
            
            # To speed up, we can pre-calculate available degrees for host vertices in remaining_edges.
            # But remaining_edges changes.
            
            # Let's use a recursive helper.
            # mapping: list of length k, mapping[i] = host vertex for tree vertex i.
            # visited_host: set of host vertices already used.
            
            def dfs_tree(node, parent, mapping, visited_host):
                if time.time() > deadline:
                    return False
                
                if len(mapping) == k:
                    return True
                
                # Get neighbors of node in tree
                # Sort neighbors by degree (desc) to fail fast?
                # Or just iterate.
                neighbors = [n for n in adj[node] if n != parent]
                
                # Heuristic: Try neighbors with higher degree first
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                
                for neighbor in neighbors:
                    if neighbor in mapping:
                        continue # Should not happen if we only visit unvisited tree nodes
                    
                    # Try to assign a host vertex to 'neighbor'
                    # The host vertex must be:
                    # 1. Not in visited_host
                    # 2. Connected to mapping[node] in remaining_edges
                    
                    # Get candidates for neighbor
                    # Candidates are host vertices h such that (mapping[node], h) in remaining_edges and h not in visited_host
                    current_host = mapping[node]
                    candidates = [h for h in range(20) if h not in visited_host and (min(current_host, h), max(current_host, h)) in remaining_edges]
                    
                    # Sort candidates? Maybe by remaining degree?
                    # For now, just iterate.
                    
                    for h in candidates:
                        mapping[neighbor] = h
                        visited_host.add(h)
                        
                        if dfs_tree(neighbor, node, mapping, visited_host):
                            return True
                        
                        mapping[neighbor] = -1 # Reset
                        visited_host.remove(h)
                
                return False

            # Iterate over possible host vertices for the root
            for root_host in range(20):
                mapping = [-1] * k
                mapping[root] = root_host
                visited_host = {root_host}
                
                if dfs_tree(root, -1, mapping, visited_host):
                    # Found a valid embedding
                    # Consume the edges
                    for u, v in tree_edges:
                        e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                        remaining_edges.remove(e)
                    
                    results[orig_idx] = mapping
                    if backtrack(idx + 1):
                        return True
                    
                    # Backtrack: Restore edges
                    for u, v in tree_edges:
                        e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                        remaining_edges.add(e)
                    results[orig_idx] = None
                    continue
                else:
                    # No embedding found with this root_host
                    pass
            
            return False

        if backtrack(0):
            return results
        return None

    # Try multiple random restarts if the first attempt fails
    # Although the order is deterministic (size desc), we can shuffle trees of same size
    # or just try again with a different internal random seed if we had randomization in candidate selection.
    # Currently, candidate selection is deterministic. 
    # To allow restarts, we should randomize the order of candidates in dfs_tree.
    
    # Let's modify dfs_tree to use a randomized order if we want restarts to be effective.
    # But for now, let's just try once. If it fails, we return None (score 0).
    # Given the perfect packing nature, a simple backtracker might get stuck.
    # Let's add a simple randomization to the candidate list in dfs_tree to enable restarts.
    
    # I will rewrite the inner logic to support randomization.
    
    def try_pack_randomized(seed_offset=0):
        remaining_edges = set(all_edges)
        results = [None] * 20
        rng = Random(12345 + seed_offset)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(indexed_trees):
                return True
            
            _, orig_idx, tree_edges = indexed_trees[idx]
            k = len(tree_edges) + 1 if tree_edges else 1
            
            if k == 1:
                results[orig_idx] = [0]
                if backtrack(idx + 1):
                    return True
                return False

            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            root = max(range(k), key=lambda x: len(adj[x]))
            
            def dfs_tree(node, parent, mapping, visited_host):
                if time.time() > deadline:
                    return False
                
                if len(mapping) == k:
                    return True
                
                neighbors = [n for n in adj[node] if n != parent]
                # Randomize neighbor order?
                # rng.shuffle(neighbors) # Might be expensive to call rng in deep recursion
                # Let's just sort by degree for now, and rely on restarts for candidate order.
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                
                for neighbor in neighbors:
                    if neighbor in mapping:
                        continue
                    
                    current_host = mapping[node]
                    candidates = [h for h in range(20) if h not in visited_host and (min(current_host, h), max(current_host, h)) in remaining_edges]
                    
                    # Randomize candidates
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        mapping[neighbor] = h
                        visited_host.add(h)
                        
                        if dfs_tree(neighbor, node, mapping, visited_host):
                            return True
                        
                        mapping[neighbor] = -1
                        visited_host.remove(h)
                
                return False

            for root_host in range(20):
                mapping = [-1] * k
                mapping[root] = root_host
                visited_host = {root_host}
                
                if dfs_tree(root, -1, mapping, visited_host):
                    for u, v in tree_edges:
                        e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                        remaining_edges.remove(e)
                    
                    results[orig_idx] = mapping
                    if backtrack(idx + 1):
                        return True
                    
                    for u, v in tree_edges:
                        e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                        remaining_edges.add(e)
                    results[orig_idx] = None
                    continue
            
            return False

        if backtrack(0):
            return results
        return None

    # Try a few restarts
    for i in range(100):
        res = try_pack_randomized(i)
        if res is not None:
            return res
            
    # If all fail, return None
    return None
