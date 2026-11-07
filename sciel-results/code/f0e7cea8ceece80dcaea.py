import time
import random
import itertools

# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Pre-compute edge set for quick lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    best_result = None
    best_count = float('inf')
    
    # Run multiple random restarts
    for seed in range(20):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # 1. Greedy Partition
        result = greedy_partition(n, edge_set, deadline, seed)
        
        if result is None:
            continue
            
        # 2. Local Search Improvement
        result = local_search(n, edge_set, result, deadline)
        
        # 3. Validate
        if validate_partition(n, edge_set, result):
            count = len(result)
            if count < best_count:
                best_count = count
                best_result = result
                
        # If we found a very good solution, we might stop early
        if best_count <= 10: 
             # Heuristic: if we get very low, further improvements are hard
             # But let's keep searching a bit more if time permits
             pass

    if best_result is None:
        # Fallback: trivial partition (each edge is a biclique)
        return [[[u], [v]] for u, v in edges]
        
    return best_result

def greedy_partition(n, edge_set, deadline, seed):
    remaining_edges = set(edge_set)
    bicliques = []
    
    # Build adjacency for quick neighbor checks
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining_edges:
        if time.time() > deadline:
            return None # Timeout, discard this attempt
            
        # Select an edge to start the biclique
        # Heuristic: Pick an edge with high degree in the remaining graph
        # To speed up, we don't calculate full degrees, just pick one randomly 
        # or based on a simple metric. 
        # For better performance, let's pick the edge with max (deg(u)+deg(v)) in remaining graph
        # But calculating degrees for all edges is O(E). 
        # Let's just pick a random edge from remaining_edges to allow variation,
        # but try to pick "dense" areas.
        
        # A simple heuristic: Pick an edge, then expand.
        # To improve, we can try a few candidate edges and pick the one that yields the largest biclique.
        
        candidates = list(remaining_edges)
        random.shuffle(candidates)
        
        best_biclique = None
        best_size = -1
        
        # Try top 5 candidates
        for edge in candidates[:5]:
            u, v = edge
            # Find maximal biclique containing (u, v) in remaining_edges
            b_l, b_r = expand_biclique(u, v, remaining_edges, adj, n)
            size = len(b_l) * len(b_r)
            if size > best_size:
                best_size = size
                best_biclique = (b_l, b_r)
                
        if best_biclique is None:
             # Should not happen if remaining_edges is not empty
             break
             
        b_l, b_r = best_biclique
        bicliques.append([sorted(b_l), sorted(b_r)])
        
        # Remove edges covered by this biclique
        for u in b_l:
            for v in b_r:
                key = (min(u, v), max(u, v))
                if key in remaining_edges:
                    remaining_edges.remove(key)
                    
    return bicliques

def expand_biclique(u, v, remaining_edges, adj, n):
    L = {u}
    R = {v}
    
    changed = True
    while changed:
        changed = False
        
        # Try to add to L
        # A node x can be added to L if it is connected to all nodes in R
        # and is not in R
        potential_L = set()
        for r in R:
            potential_L.update(adj[r])
            
        for x in potential_L:
            if x in R: continue
            if x in L: continue
            
            # Check if x is connected to all in R via remaining_edges
            valid = True
            for r in R:
                key = (min(x, r), max(x, r))
                if key not in remaining_edges:
                    valid = False
                    break
            if valid:
                L.add(x)
                changed = True
                
        # Try to add to R
        potential_R = set()
        for l in L:
            potential_R.update(adj[l])
            
        for y in potential_R:
            if y in L: continue
            if y in R: continue
            
            # Check if y is connected to all in L via remaining_edges
            valid = True
            for l in L:
                key = (min(y, l), max(y, l))
                if key not in remaining_edges:
                    valid = False
                    break
            if valid:
                R.add(y)
                changed = True
                
    return L, R

def local_search(n, edge_set, bicliques, deadline):
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try to merge any two bicliques
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                    
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Check if union (L1 U L2, R1 U R2) is a valid biclique
                # This requires all edges between (L1 U L2) and (R1 U R2) to exist in the graph
                # AND to be covered by the union of edges of biclique i and j?
                # No, the union must be a biclique in the original graph.
                # If it is a biclique in the original graph, then all edges between L_union and R_union
                # must be in the graph. Since biclique i and j are valid, edges within L1-R1 and L2-R2 exist.
                # We need to check cross edges: L1-R2 and L2-R1.
                
                valid_merge = True
                # Check L1 vs R2
                for u in L1:
                    for v in R2:
                        key = (min(u, v), max(u, v))
                        if key not in edge_set:
                            valid_merge = False
                            break
                    if not valid_merge: break
                if not valid_merge:
                    continue
                    
                # Check L2 vs R1
                for u in L2:
                    for v in R1:
                        key = (min(u, v), max(u, v))
                        if key not in edge_set:
                            valid_merge = False
                            break
                    if not valid_merge: break
                if not valid_merge:
                    continue
                    
                # If valid, merge them
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                # Update bicliques list
                # Remove j and i, add new one
                bicliques.pop(j)
                bicliques.pop(i)
                bicliques.append([sorted(new_L), sorted(new_R)])
                improved = True
                break # Restart the outer loop since indices changed
            if improved:
                break
                
    return bicliques

def validate_partition(n, edge_set, bicliques):
    if not bicliques:
        return len(edge_set) == 0
        
    covered_edges = set()
    
    for L, R in bicliques:
        # Check non-empty and disjoint
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
            
        # Check all cross edges exist in graph and are added to covered
        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    return False
                if key in covered_edges:
                    return False # Duplicate coverage
                covered_edges.add(key)
                
    # Check all edges are covered
    if covered_edges != edge_set:
        return False
        
    return True
# EVOLVE-BLOCK-END
