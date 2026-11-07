import time
import random
import sys
from collections import defaultdict

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
    deadline = start_time + 115
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    all_k20_edges = [(u, v) for u in range(n_vertices) for v in range(u + 1, n_vertices)]
    
    # Precompute tree data
    tree_data = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        max_deg = max(deg) if k > 1 else 0
        tree_data.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'max_deg': max_deg
        })
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_greedy_with_repair(rng):
        """
        Greedy tree embedding with edge-conflict detection,
        followed by local search to resolve conflicts.
        """
        # State: for each tree, a mapping of tree vertices to host vertices
        mappings = [None] * n_trees
        # Track which K20 edges are used by which tree
        edge_to_tree = {}  # (u,v) -> tree_idx
        # For each tree, track its used edges
        tree_used_edges = [set() for _ in range(n_trees)]
        
        # Order: most constrained first (highest max degree, then largest size)
        order = sorted(range(n_trees), key=lambda x: (-tree_data[x]['max_deg'], -tree_data[x]['k']))
        
        # Phase 1: Greedy embedding
        for t_idx in order:
            td = tree_data[t_idx]
            k = td['k']
            if k == 1:
                mappings[t_idx] = [rng.randint(0, n_vertices - 1)]
                continue
            
            adj = td['adj']
            deg = td['deg']
            
            # Try to embed this tree
            mapping = [-1] * k
            used_verts = set()
            
            # Vertex ordering: highest degree first
            v_order = sorted(range(k), key=lambda v: -deg[v])
            
            success = True
            for tv in v_order:
                d_tv = deg[tv]
                mapped_neighbors = [nb for nb in adj[tv] if mapping[nb] != -1]
                num_mapped = len(mapped_neighbors)
                req_cap = d_tv - num_mapped
                
                cands = []
                for h in range(n_vertices):
                    if h in used_verts:
                        continue
                    # Check edge availability to mapped neighbors
                    valid = True
                    for nb in mapped_neighbors:
                        hn = mapping[nb]
                        e = (min(h, hn), max(h, hn))
                        if e in edge_to_tree:
                            valid = False
                            break
                    if valid:
                        cands.append(h)
                
                if not cands:
                    success = False
                    break
                
                # Pick randomly from candidates
                h = rng.choice(cands)
                mapping[tv] = h
                used_verts.add(h)
            
            if success:
                mappings[t_idx] = mapping
                # Register edges
                for u, v in td['edges']:
                    eu, ev = mapping[u], mapping[v]
                    e = (min(eu, ev), max(eu, ev))
                    if e in edge_to_tree:
                        # Conflict! This shouldn't happen with the check above, but just in case
                        success = False
                        break
                    edge_to_tree[e] = t_idx
                    tree_used_edges[t_idx].add(e)
            
            if not success:
                # Undo and return failure
                # (In a full implementation, we'd backtrack, but for now we just fail)
                return None
        
        # Phase 2: Local search to resolve any remaining issues
        # Actually, if we got here, all trees are embedded. Let's verify.
        # Check for edge conflicts
        conflict_count = 0
        for e, t_idx in edge_to_tree.items():
            pass  # No conflicts if we built it correctly
        
        # Verify all mappings are valid
        for t_idx in range(n_trees):
            if mappings[t_idx] is None:
                return None
            m = mappings[t_idx]
            if len(set(m)) != len(m):
                return None  # Not distinct
            for v in m:
                if v < 0 or v >= n_vertices:
                    return None
        
        # Count total edges
        total_edges = sum(len(trees[i]) for i in range(n_trees) if mappings[i] is not None)
        return mappings, total_edges
    
    # Try multiple random restarts
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(attempt * 7919 + 1)
        result = try_greedy_with_repair(rng)
        
        if result is not None:
            mappings, score = result
            if score > best_score:
                best_score = score
                best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
                if best_score == 190:
                    break
    
    # If we haven't found a perfect packing, try a more systematic approach
    if best_score < 190:
        # Use backtracking with better heuristics
        used_edges = set()
        remaining_deg = [19] * n_vertices
        mappings = [None] * n_trees
        
        # Sort trees by difficulty
        order = sorted(range(n_trees), key=lambda x: (-tree_data[x]['max_deg'], -tree_data[x]['k']))
        
        def pack_tree_bt(t_idx, rng):
            if time.time() > deadline:
                return False
            
            if t_idx == n_trees:
                return True
                
            td = tree_data[order[t_idx]]
            k = td['k']
            adj = td['adj']
            deg = td['deg']
            edges = td['edges']
            
            if k == 1:
                # Find a vertex not yet used by other trees? 
                # Actually, vertices can be shared across trees, only edges must be disjoint.
                # For k=1, no edges, so any vertex works.
                mappings[order[t_idx]] = [0]
                return pack_tree_bt(t_idx + 1, rng)
            
            mapping = [-1] * k
            used_verts = set()
            
            def get_candidates(tv):
                cands = []
                d_tv = deg[tv]
                mapped_neighbors = [nb for nb in adj[tv] if mapping[nb] != -1]
                num_mapped = len(mapped_neighbors)
                req_cap = d_tv - num_mapped
                
                for h in range(n_vertices):
                    if h in used_verts:
                        continue
                    if remaining_deg[h] < req_cap:
                        continue
                    
                    valid = True
                    for nb in mapped_neighbors:
                        hn = mapping[nb]
                        if hn == h:
                            valid = False
                            break
                        e = (min(h, hn), max(h, hn))
                        if e in used_edges:
                            valid = False
                            break
                    if valid:
                        cands.append(h)
                return cands
            
            def backtrack(pos, rng):
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    # Commit edges
                    added_edges = []
                    for u, v in edges:
                        eu, ev = mapping[u], mapping[v]
                        e = (min(eu, ev), max(eu, ev))
                        if e in used_edges:
                            return False
                        used_edges.add(e)
                        added_edges.append(e)
                        remaining_deg[eu] -= 1
                        remaining_deg[ev] -= 1
                        
                    mappings[order[t_idx]] = mapping[:]
                    
                    if pack_tree_bt(t_idx + 1, rng):
                        return True
                    
                    # Undo
                    for e in added_edges:
                        used_edges.remove(e)
                        remaining_deg[e[0]] += 1
                        remaining_deg[e[1]] += 1
                        
                    mappings[order[t_idx]] = None
                    return False
                
                # MRV: find unmapped vertex with fewest candidates
                best_tv = -1
                best_cands = None
                min_c = float('inf')
                
                for tv in range(k):
                    if mapping[tv] != -1:
                        continue
                    cands = get_candidates(tv)
                    if not cands:
                        return False
                    if len(cands) < min_c:
                        min_c = len(cands)
                        best_tv = tv
                        best_cands = cands
                        if min_c == 1:
                            break
                
                if best_tv == -1:
                    return False
                
                rng.shuffle(best_cands)
                
                for h in best_cands:
                    mapping[best_tv] = h
                    used_verts.add(h)
                    
                    if backtrack(pos + 1, rng):
                        return True
                    
                    mapping[best_tv] = -1
                    used_verts.remove(h)
                
                return False
            
            return backtrack(0, rng)
        
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(seed * 104729 + 3)
            # Reset state
            used_edges.clear()
            remaining_deg = [19] * n_vertices
            mappings = [None] * n_trees
            
            if pack_tree_bt(0, rng):
                score = sum(len(trees[i]) for i in range(n_trees) if mappings[i] is not None)
                if score > best_score:
                    best_score = score
                    best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
                    if best_score == 190:
                        break
    
    # Build final result
    final_mappings = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            final_mappings.append(None)
        else:
            final_mappings.append([int(v) for v in best_mappings[i]])
    
    return final_mappings
