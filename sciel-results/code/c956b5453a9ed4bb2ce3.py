import time
import random
from random import Random

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

def construct():
    start_time = time.time()
    # 150s limit, keep 10s margin
    deadline = start_time + 140.0
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency and degrees for each tree
    tree_adj = []
    tree_degrees = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_degrees.append([len(a) for a in adj])
    
    # Sort trees by size descending (Largest First)
    # This is crucial: place the most constrained (largest) trees first
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    # Precompute all edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # We will try multiple random restarts
    # Since K20 is dense and trees are small, a solution is likely found quickly
    max_restarts = 10000 
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 7919 + 13)
        
        # Track available edges
        available_edges = set(all_edges)
        # Track used vertices in K20? 
        # Note: Vertices can be shared between different trees?
        # The problem says "Embedded trees must be edge-disjoint".
        # It does NOT say vertex-disjoint.
        # However, the return format is "list of k distinct int host vertices".
        # This implies that within a single tree, vertices must be distinct.
        # But across trees, vertices CAN be reused.
        # Let's re-read carefully: "Entry k-1 is None or a list of k distinct int host vertices... Embedded trees must be edge-disjoint in K20."
        # Yes, vertex reuse is allowed across trees.
        
        solution = [None] * n
        success = True
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: Single vertex
                solution[t_idx] = [rng.randint(0, n - 1)]
                continue
            
            adj = tree_adj[t_idx]
            degs = tree_degrees[t_idx]
            
            # Root selection: Pick a node with high degree to start?
            # Or random? Random is safer for restarts.
            # Let's pick a random root.
            root = rng.randint(0, k - 1)
            
            # DFS state
            mapping = [-1] * k
            used_in_tree = set()
            
            # Helper to get available neighbors of a host vertex
            # We need to check if edge (host, v) is in available_edges
            # And v is not in used_in_tree
            
            def get_candidates(parent_host):
                candidates = []
                for v in range(n):
                    if v in used_in_tree:
                        continue
                    if parent_host == -1:
                        candidates.append(v)
                    else:
                        e = (min(parent_host, v), max(parent_host, v))
                        if e in available_edges:
                            candidates.append(v)
                return candidates
            
            # Heuristic for sorting candidates:
            # Prefer vertices with more available edges (higher degree in remaining graph)
            # This leaves more options for the rest of the tree.
            def sort_candidates(cands, node_deg):
                if len(cands) <= 1:
                    return cands
                
                # Calculate remaining degree for each candidate
                # This is O(N) per candidate, so O(N^2) per node.
                # N=20, so 400 ops. Acceptable.
                scored = []
                for v in cands:
                    rem_deg = 0
                    for w in range(n):
                        if w == v or w in used_in_tree:
                            continue
                        e = (min(v, w), max(v, w))
                        if e in available_edges:
                            rem_deg += 1
                    scored.append((rem_deg, v))
                
                # Sort by remaining degree descending
                # If node_deg is high, we really want high rem_deg.
                # If node_deg is low (leaf), it matters less, but still good.
                scored.sort(key=lambda x: x[0], reverse=True)
                return [v for _, v in scored]

            def dfs(node, parent_host):
                if time.time() > deadline:
                    return False
                
                cands = get_candidates(parent_host)
                if not cands:
                    return False
                
                cands = sort_candidates(cands, degs[node])
                
                for gv in cands:
                    mapping[node] = gv
                    used_in_tree.add(gv)
                    
                    ok = True
                    # Process children
                    # Order children by degree descending to fail fast?
                    children = [nb for nb in adj[node] if mapping[nb] == -1]
                    # Sort children by degree in tree, descending
                    children.sort(key=lambda x: degs[x], reverse=True)
                    
                    for nb in children:
                        if not dfs(nb, gv):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    # Backtrack
                    used_in_tree.remove(gv)
                    mapping[node] = -1
                
                return False
            
            if not dfs(root, -1):
                success = False
                break
            
            # If successful, mark edges as used
            for u, v in trees[t_idx]:
                gu = mapping[u]
                gv = mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available_edges:
                    available_edges.remove(e)
            
            solution[t_idx] = mapping[:]
        
        if success:
            # Verify and return
            # The problem asks for a list of 20 entries.
            # Entry k-1 corresponds to T_k.
            # Our solution list is indexed by t_idx which is 0..19 corresponding to T_1..T_20.
            return solution
    
    # Fallback if no complete solution found in time
    # Try to return whatever we have, or a trivial solution
    # Since we break on failure, we might not have a full solution.
    # Let's try a greedy approach without backtracking for a fallback?
    # Or just return None for all?
    # Better: Return the best partial solution found in the last few restarts?
    # Given the high probability of success, this is unlikely to be hit.
    # But to be safe, let's construct a valid partial packing.
    
    # Simple greedy fallback
    result = [None] * n
    available_edges_fb = set(all_edges)
    
    for t_idx in order:
        k = t_idx + 1
        if k == 1:
            result[t_idx] = [0]
            continue
        
        adj = tree_adj[t_idx]
        degs = tree_degrees[t_idx]
        root = 0
        mapping = [-1] * k
        used_in_tree = set()
        
        # Simple DFS without heavy heuristics, just to get *some* embedding
        def dfs_fb(node, parent_host):
            cands = []
            for v in range(n):
                if v in used_in_tree:
                    continue
                if parent_host == -1:
                    cands.append(v)
                else:
                    e = (min(parent_host, v), max(parent_host, v))
                    if e in available_edges_fb:
                        cands.append(v)
            
            if not cands:
                return False
            
            for gv in cands:
                mapping[node] = gv
                used_in_tree.add(gv)
                
                ok = True
                for nb in adj[node]:
                    if mapping[nb] == -1:
                        if not dfs_fb(nb, gv):
                            ok = False
                            break
                
                if ok:
                    return True
                
                used_in_tree.remove(gv)
                mapping[node] = -1
            
            return False
        
        if dfs_fb(root, -1):
            for u, v in trees[t_idx]:
                gu = mapping[u]
                gv = mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available_edges_fb:
                    available_edges_fb.remove(e)
            result[t_idx] = mapping[:]
    
    return result

# Ensure Random is available
from random import Random
