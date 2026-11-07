"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    
    # Build adjacency and degree for each tree
    tree_adj = []
    tree_max_deg = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_max_deg.append(max(len(adj[i]) for i in range(k)) if k > 1 else 0)
        
    best_result = [None] * 20
    best_edges = 0
    
    # Precompute all edges in K20
    all_edges = set()
    for a in range(20):
        for b in range(a + 1, 20):
            all_edges.add((a, b))
    
    # For each tree, precompute the edges it needs
    tree_edges_list = []
    for k in range(20):
        tree_edges_list.append(trees[k])
    
    # CSP-based packing: treat tree edges as variables, K20 edges as values
    def try_pack_csp(order, rng):
        """
        CSP approach: 
        - Variables: tree edges (across all trees in order)
        - Values: K20 edges (a,b) where a<b
        - Constraints: 
          1. Edge-disjointness: no two tree edges map to same K20 edge
          2. Vertex consistency: if tree edge (u,v) maps to (a,b), then all edges incident to u in the tree must map to edges incident to a
        """
        used_edges = set()
        result = [None] * 20
        # Track vertex mappings per tree
        # For each tree, we need consistent vertex mappings
        # This is tricky in pure CSP... let me think differently.
        
        # Actually, let's use a hybrid: for each tree, do backtracking with
        # dynamic variable ordering (most constrained vertex first) and
        # value ordering (least constraining host vertex first)
        
        for k_idx in order:
            k = k_idx + 1
            if k == 1:
                result[k_idx] = [0]
                continue
                
            if k == 2:
                found = False
                for a in range(20):
                    for b in range(a + 1, 20):
                        edge = (a, b)
                        if edge not in used_edges:
                            used_edges.add(edge)
                            result[k_idx] = [a, b]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue
                continue
            
            adj = tree_adj[k_idx]
            k_val = k
            mapping = [-1] * k_val
            host_used = [False] * 20
            
            # Compute degrees
            deg = [len(adj[i]) for i in range(k_val)]
            
            # Dynamic variable ordering: most constrained vertex first
            # A vertex is "constrained" by: its degree (more neighbors = more constraints)
            # and by how many of its neighbors are already assigned
            
            def get_var_order(assigned_count):
                """Get variable ordering based on current state"""
                # Score: prefer vertices with more assigned neighbors (more constrained)
                # and higher degree
                scores = []
                for i in range(k_val):
                    if mapping[i] != -1:
                        continue
                    assigned_neighbors = sum(1 for nb in adj[i] if mapping[nb] != -1)
                    # Higher score = more constrained = assign first
                    score = assigned_neighbors * 10 + deg[i]
                    scores.append((score, i))
                scores.sort(reverse=True)
                return [s[1] for s in scores]
            
            def backtrack_csp():
                if time.time() > deadline:
                    return False
                
                var_order = get_var_order(sum(1 for m in mapping if m != -1))
                if not var_order:
                    return True
                
                tv = var_order[0]  # Most constrained vertex
                
                # Value ordering: least constraining value first
                # For each candidate host, count how many future edges it would block
                candidates = []
                for h in range(20):
                    if host_used[h]:
                        continue
                    # Check edges to assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if mapping[nb] != -1:
                            hnb = mapping[nb]
                            edge = (min(h, hnb), max(h, hnb))
                            if edge in used_edges:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    # Count how many free edges from h would be "wasted" 
                    # (i.e., edges from h to free hosts that are already used)
                    # Actually, LCV: prefer values that leave most options for others
                    # Count available edges from h to unassigned tree vertices
                    avail_count = 0
                    for nb in adj[tv]:
                        if mapping[nb] == -1:
                            hnb = h  # This doesn't make sense...
                            # For each unassigned neighbor, how many free edges from h?
                            # We can't know yet. Let's use a simpler heuristic.
                            pass
                    
                    # Simpler LCV: prefer hosts with fewer used edges incident
                    # (more free space)
                    used_incident = 0
                    for (a, b) in used_edges:
                        if a == h or b == h:
                            used_incident += 1
                    candidates.append((used_incident, h))
                
                # Sort by used_incident ascending (prefer hosts with more free space)
                candidates.sort(key=lambda x: x[0])
                
                for _, hv in candidates:
                    # Assign tentatively
                    mapping[tv] = hv
                    host_used[hv] = True
                    
                    # Forward checking: check if any unassigned neighbor is now blocked
                    fc_ok = True
                    for nb in adj[tv]:
                        if mapping[nb] == -1:
                            # Check if nb has any available host
                            has_host = False
                            for h in range(20):
                                if host_used[h]:
                                    continue
                                ok = True
                                for nb2 in adj[nb]:
                                    if mapping[nb2] != -1:
                                        hnb2 = mapping[nb2]
                                        edge = (min(h, hnb2), max(h, hnb2))
                                        if edge in used_edges:
                                            ok = False
                                            break
                                if ok:
                                    has_host = True
                                    break
                            if not has_host:
                                fc_ok = False
                                break
                    
                    if fc_ok:
                        if backtrack_csp():
                            return True
                    
                    # Undo
                    mapping[tv] = -1
                    host_used[hv] = False
                
                return False
            
            if backtrack_csp():
                result[k_idx] = [mapping[i] for i in range(k_val)]
                for (u, v) in trees[k_idx]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                pass
                
        return result, used_edges
    
    # Strategy: Try different orderings with the CSP solver
    # Sort by complexity first
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-tree_max_deg[i], -(i+1)))
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 7919 + 12345)
        
        # Create a randomized order based on complexity
        groups = {}
        for i in tree_indices:
            md = tree_max_deg[i]
            if md not in groups:
                groups[md] = []
            groups[md].append(i)
            
        sorted_keys = sorted(groups.keys(), reverse=True)
        current_order = []
        for key in sorted_keys:
            grp = groups[key]
            rng.shuffle(grp)
            current_order.extend(grp)
            
        res, used = try_pack_csp(current_order, rng)
        
        total_edges = sum(len(trees[k]) for k in range(20) if res[k] is not None)
        
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = res[:]
            if best_edges == 190:
                break
                
    # If we didn't get 190, try random orderings
    if best_edges < 190:
        for attempt in range(1000):
            if time.time() > deadline:
                break
            rng = Random(attempt * 104729 + 67890)
            current_order = list(range(20))
            rng.shuffle(current_order)
            
            res, used = try_pack_csp(current_order, rng)
            
            total_edges = sum(len(trees[k]) for k in range(20) if res[k] is not None)
            
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = res[:]
                if best_edges == 190:
                    break

    final_result = []
    for k in range(20):
        if best_result[k] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[k]])
            
    return final_result
# EVOLVE-BLOCK-END
