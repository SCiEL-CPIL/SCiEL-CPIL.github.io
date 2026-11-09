import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 145.0  # Safety margin before 150s kill

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _is_clique(vertices, adj):
    """Check if a set of vertices forms a clique in the graph defined by adj."""
    v_list = list(vertices)
    for i in range(len(v_list)):
        for j in range(i + 1, len(v_list)):
            if v_list[j] not in adj[v_list[i]]:
                return False
    return True

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency sets for fast intersection and lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    initial_edges_set = set(edges)
    
    best_partition = None
    best_count = float('inf')
    
    # Number of restarts. We will adjust based on time if possible, 
    # but fixed 30 is a good balance for N=64.
    num_restarts = 30
    
    for run_seed in range(num_restarts):
        if not _deadline_ok():
            break
            
        current_edges = set(initial_edges_set)
        partition = []
        rng = random.Random(seed + run_seed)
        
        # Greedy Peeling
        while current_edges:
            if not _deadline_ok():
                # Fallback: add remaining edges as K2
                for e in current_edges:
                    partition.append([e[0], e[1]])
                break
            
            # 1. Find edge with minimum common neighbors in the REMAINING graph
            # To be efficient, we iterate through edges and compute intersection size.
            # For N=64, E is ~2000. Intersection of sets of size ~30 is fast.
            min_common = float('inf')
            candidates = []
            
            # Optimization: We can sample a subset of edges if time is tight, 
            # but for N=64, full scan is usually fast enough in Python sets.
            for u, v in current_edges:
                # Common neighbors of u and v in the remaining graph
                # w is common if w in adj[u] and w in adj[v] 
                # AND (u,w) in current_edges AND (v,w) in current_edges
                # Note: adj[u] contains original neighbors. We must filter by current_edges.
                # However, if (u,v) is in current_edges, and w is in adj[u] \cap adj[v],
                # we still need to check if (u,w) and (v,w) are in current_edges.
                
                # A faster way: 
                # The common neighbors in the remaining graph are 
                # (adj[u] & adj[v]) intersected with vertices connected to both u and v in current graph.
                # Actually, if we maintain adj dynamically, it's easier. 
                # But maintaining adj dynamically is expensive (removing edges).
                # Let's stick to checking membership in current_edges.
                
                # To speed up, we can precompute the set of neighbors in current graph? No, that's dynamic.
                # Let's just do the intersection of original adj, then filter.
                common_set = adj[u] & adj[v]
                common_count = 0
                for w in common_set:
                    if w == u or w == v:
                        continue
                    e_uw = (u, w) if u < w else (w, u)
                    e_vw = (v, w) if v < w else (w, v)
                    if e_uw in current_edges and e_vw in current_edges:
                        common_count += 1
                
                if common_count < min_common:
                    min_common = common_count
                    candidates = [(u, v)]
                elif common_count == min_common:
                    candidates.append((u, v))
            
            if not candidates:
                # Should not happen
                break
                
            # Pick a candidate edge
            u, v = rng.choice(candidates)
            
            # 2. Expand to maximal clique
            clique = [u, v]
            
            # Candidates for expansion: common neighbors of u and v in current graph
            candidates_w = []
            common_set = adj[u] & adj[v]
            for w in common_set:
                if w == u or w == v:
                    continue
                e_uw = (u, w) if u < w else (w, u)
                e_vw = (v, w) if v < w else (w, v)
                if e_uw in current_edges and e_vw in current_edges:
                    candidates_w.append(w)
            
            # Greedy expansion: add vertex with max connections to current clique
            # To make it more robust, we can try a few different expansion orders?
            # The current approach picks max degree in candidate set.
            
            while candidates_w:
                if not _deadline_ok():
                    break
                
                # Pick w with max number of neighbors in candidates_w within current_edges
                max_deg = -1
                best_w = -1
                tie_list = []
                
                for w in candidates_w:
                    deg = 0
                    # Count neighbors of w in candidates_w that are connected in current graph
                    for x in candidates_w:
                        if w == x:
                            continue
                        e_wx = (w, x) if w < x else (x, w)
                        if e_wx in current_edges:
                            deg += 1
                    
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                        tie_list = [w]
                    elif deg == max_deg:
                        tie_list.append(w)
                
                if best_w == -1:
                    break
                    
                # If ties, pick randomly
                if len(tie_list) > 1:
                    best_w = rng.choice(tie_list)
                    
                clique.append(best_w)
                candidates_w.remove(best_w)
                
                # Filter candidates_w: must be connected to best_w in current graph
                new_candidates_w = []
                for w in candidates_w:
                    e_wbw = (w, best_w) if w < best_w else (best_w, w)
                    if e_wbw in current_edges:
                        new_candidates_w.append(w)
                candidates_w = new_candidates_w
                
            partition.append(clique)
            
            # 3. Remove edges of the clique from current_edges
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                if e in current_edges:
                    current_edges.remove(e)
                    
        # 4. Post-processing: Merge cliques if possible
        # Iterate until no more merges can be done
        merged = True
        while merged:
            if not _deadline_ok():
                break
            merged = False
            new_partition = []
            i = 0
            while i < len(partition):
                C1 = set(partition[i])
                found_merge = False
                for j in range(i + 1, len(partition)):
                    if not _deadline_ok():
                        break
                    C2 = set(partition[j])
                    union = C1 | C2
                    # Check if union is a clique
                    if _is_clique(union, adj):
                        # Merge
                        new_partition.append(list(union))
                        # Mark j as removed
                        partition[j] = None
                        found_merge = True
                        merged = True
                        break
                if not found_merge:
                    new_partition.append(partition[i])
                i += 1
            
            # Filter out Nones
            new_partition = [c for c in new_partition if c is not None]
            
            if len(new_partition) < len(partition):
                partition = new_partition
            else:
                merged = False

        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = partition
            
    if best_partition is None:
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
