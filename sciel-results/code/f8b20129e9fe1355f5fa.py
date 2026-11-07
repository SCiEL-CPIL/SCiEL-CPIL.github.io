import time
import random
import sys

sys.setrecursionlimit(100000)

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
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
    # Precompute tree adjacencies and edges
    tree_adj = []
    tree_edges = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)
    
    # Sort trees by size descending (larger trees first for better backtracking)
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: -(x + 1))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees
        
        # For each tree, we'll try to pack it using backtracking
        for order_idx in range(n_trees):
            if time.time() > deadline:
                break
            
            t_idx = tree_indices[order_idx]
            k = t_idx + 1
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            
            if k == 1:
                mappings[t_idx] = [0]
                continue
            
            # Backtracking search for this tree
            mapping = [-1] * k
            used_verts = set()
            nodes_visited = 0
            max_nodes = 500000  # Increased limit
            
            def get_candidates(tv):
                """Get valid host vertices for tree vertex tv"""
                candidates = []
                for hv in range(n_vertices):
                    if hv in used_verts:
                        continue
                    valid = True
                    for nb in adj_t[tv]:
                        if mapping[nb] != -1:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges:
                                valid = False
                                break
                    if valid:
                        candidates.append(hv)
                return candidates
            
            def backtrack(pos):
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > max_nodes or time.time() > deadline:
                    return False
                
                if pos == k:
                    # Commit: add all tree edges to used_edges
                    for u, v in edges_t:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    mappings[t_idx] = mapping[:]
                    return True
                
                # MCV: Pick unmapped vertex with fewest valid host candidates
                best_tv = -1
                best_candidates = None
                min_cand_count = float('inf')
                
                for tv in range(k):
                    if mapping[tv] != -1:
                        continue
                    
                    candidates = get_candidates(tv)
                    if not candidates:
                        return False
                    
                    if len(candidates) < min_cand_count:
                        min_cand_count = len(candidates)
                        best_tv = tv
                        best_candidates = candidates
                        if min_cand_count == 1:
                            break
                
                if best_tv == -1:
                    return False
                
                # Order candidates: prefer those with fewer conflicts to already-mapped neighbors
                # and more flexibility for future vertices
                def candidate_score(hv):
                    score = 0
                    # Count edges from hv to already-mapped vertices that are in used_edges
                    # (this is 0 by construction since we filtered, but count adjacent used edges)
                    for u, v in edges_t:
                        if mapping[u] != -1 and mapping[v] == -1:
                            if (min(hv, mapping[u]), max(hv, mapping[u])) in used_edges:
                                score += 1
                        elif mapping[v] != -1 and mapping[u] == -1:
                            if (min(hv, mapping[v]), max(hv, mapping[v])) in used_edges:
                                score += 1
                    return score
                
                best_candidates.sort(key=candidate_score)
                
                # Add some randomness to break ties
                if len(best_candidates) > 1 and local_rng.random() < 0.3:
                    # Swap two random elements to add diversity
                    i, j = local_rng.sample(range(len(best_candidates)), 2)
                    best_candidates[i], best_candidates[j] = best_candidates[j], best_candidates[i]
                
                for hv in best_candidates:
                    mapping[best_tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1):
                        return True
                    used_verts.remove(hv)
                    mapping[best_tv] = -1
                
                return False
            
            if not backtrack(0):
                # Failed to pack this tree, return what we have
                break
        
        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
        
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
        
        return best_score == 190
    
    # Try multiple random restarts
    attempt = 0
    while time.time() < deadline:
        if try_packing(attempt):
            break
        attempt += 1
    
    # Ensure all mappings are lists of Python ints or None
    result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    
    return result
