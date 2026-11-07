"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    total_edges = len(edge_set)
    
    def is_valid_partition(bicliques):
        """Check if bicliques form a valid partition."""
        covered = set()
        for left, right in bicliques:
            if not left or not right:
                return False
            # Check disjointness within sides
            if len(set(left)) != len(left) or len(set(right)) != len(right):
                return False
            # Check no overlap between left and right
            if set(left) & set(right):
                return False
            # Check all cross pairs are edges
            for u in left:
                for v in right:
                    if u < v:
                        pair = (u, v)
                    else:
                        pair = (v, u)
                    if pair not in edge_set:
                        return False
                    if pair in covered:
                        return False
                    covered.add(pair)
        return covered == edge_set
    
    def greedy_biclique_partition(remaining_edges, adj, n):
        """Greedy: repeatedly find largest biclique in remaining edges."""
        bicliques = []
        remaining = set(remaining_edges)
        
        while remaining:
            # Find largest biclique in remaining edges
            best_biclique = None
            best_size = 0
            
            # Try each edge as seed
            for u, v in list(remaining)[:50]:  # Limit for speed
                # Build candidate biclique starting from edge (u,v)
                left = {u}
                right = {v}
                
                # Try to expand
                changed = True
                while changed:
                    changed = False
                    # Try adding to left: vertex w adjacent to all in right
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in right):
                                # Check all edges from w to right are in remaining
                                all_in_remaining = True
                                for x in right:
                                    if w < x:
                                        if (w, x) not in remaining:
                                            all_in_remaining = False
                                            break
                                    else:
                                        if (x, w) not in remaining:
                                            all_in_remaining = False
                                            break
                                if all_in_remaining:
                                    left.add(w)
                                    changed = True
                    # Try adding to right: vertex w adjacent to all in left
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in left):
                                all_in_remaining = True
                                for x in left:
                                    if w < x:
                                        if (w, x) not in remaining:
                                            all_in_remaining = False
                                            break
                                    else:
                                        if (x, w) not in remaining:
                                            all_in_remaining = False
                                            break
                                if all_in_remaining:
                                    right.add(w)
                                    changed = True
                
                size = len(left) * len(right)
                if size > best_size:
                    best_size = size
                    best_biclique = (list(left), list(right))
            
            if best_biclique is None or best_size == 0:
                # Fallback: single edge
                u, v = next(iter(remaining))
                best_biclique = ([u], [v])
            
            bicliques.append(best_biclique)
            # Remove covered edges
            left, right = best_biclique
            for u in left:
                for v in right:
                    if u < v:
                        remaining.discard((u, v))
                    else:
                        remaining.discard((v, u))
        
        return bicliques
    
    # Initial greedy solution
    best_bicliques = greedy_biclique_partition(edge_set, adj, n)
    best_count = len(best_bicliques)
    
    # Local search: try to merge or replace bicliques
    def local_search(bicliques, adj, n, edge_set, time_limit):
        bicliques = [list(b) for b in bicliques]
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try merging pairs
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    li, ri = bicliques[i]
                    lj, rj = bicliques[j]
                    # Try merging: combine lefts and rights
                    new_left = list(set(li) | set(lj))
                    new_right = list(set(ri) | set(rj))
                    if not new_left or not new_right:
                        continue
                    if set(new_left) & set(new_right):
                        continue
                    # Check validity
                    valid = True
                    for u in new_left:
                        for v in new_right:
                            if u < v:
                                pair = (u, v)
                            else:
                                pair = (v, u)
                            if pair not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Remove i and j, add merged
                        new_bicliques = []
                        for k in range(len(bicliques)):
                            if k != i and k != j:
                                new_bicliques.append(bicliques[k])
                        new_bicliques.append([new_left, new_right])
                        if len(new_bicliques) < len(bicliques):
                            bicliques = new_bicliques
                            improved = True
                            break
                if improved:
                    break
            
            # Try replacing a biclique with two smaller ones that cover same edges
            for i in range(len(bicliques)):
                li, ri = bicliques[i]
                # Try splitting
                if len(li) > 1:
                    # Try moving some vertices from left to right
                    for u in li:
                        new_li = [x for x in li if x != u]
                        new_ri = ri + [u]
                        if set(new_li) & set(new_ri):
                            continue
                        valid1 = True
                        for a in new_li:
                            for b in new_ri:
                                if a < b:
                                    pair = (a, b)
                                else:
                                    pair = (b, a)
                                if pair not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        if valid1:
                            # Check if remaining edges can be covered
                            # This is complex; skip for now
                            pass
                if len(ri) > 1:
                    for v in ri:
                        new_ri = [x for x in ri if x != v]
                        new_li = li + [v]
                        if set(new_li) & set(new_ri):
                            continue
                        valid1 = True
                        for a in new_li:
                            for b in new_ri:
                                if a < b:
                                    pair = (a, b)
                                else:
                                    pair = (b, a)
                                if pair not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        if valid1:
                            pass
        
        return bicliques
    
    # Run local search
    time_limit = start_time + 75
    best_bicliques = local_search(best_bicliques, adj, n, edge_set, time_limit)
    best_count = len(best_bicliques)
    
    # Try multiple random restarts
    random.seed(42)
    restart_time_limit = start_time + 80
    for restart in range(20):
        if time.time() > restart_time_limit:
            break
        
        # Randomized greedy
        bicliques = []
        remaining = set(edge_set)
        
        while remaining:
            best_biclique = None
            best_size = 0
            
            candidates = list(remaining)
            random.shuffle(candidates)
            for u, v in candidates[:30]:
                left = {u}
                right = {v}
                
                changed = True
                while changed:
                    changed = False
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in right):
                                all_in_remaining = True
                                for x in right:
                                    if w < x:
                                        if (w, x) not in remaining:
                                            all_in_remaining = False
                                            break
                                    else:
                                        if (x, w) not in remaining:
                                            all_in_remaining = False
                                            break
                                if all_in_remaining:
                                    left.add(w)
                                    changed = True
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in left):
                                all_in_remaining = True
                                for x in left:
                                    if w < x:
                                        if (w, x) not in remaining:
                                            all_in_remaining = False
                                            break
                                    else:
                                        if (x, w) not in remaining:
                                            all_in_remaining = False
                                            break
                                if all_in_remaining:
                                    right.add(w)
                                    changed = True
                
                size = len(left) * len(right)
                if size > best_size:
                    best_size = size
                    best_biclique = (list(left), list(right))
            
            if best_biclique is None or best_size == 0:
                u, v = next(iter(remaining))
                best_biclique = ([u], [v])
            
            bicliques.append(best_biclique)
            left, right = best_biclique
            for u in left:
                for v in right:
                    if u < v:
                        remaining.discard((u, v))
                    else:
                        remaining.discard((v, u))
        
        # Local search on this solution
        bicliques = local_search(bicliques, adj, n, edge_set, restart_time_limit)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    # Final validation and cleanup
    if not is_valid_partition(best_bicliques):
        # Fallback to simple valid partition
        best_bicliques = [[[u], [v]] for u, v in edge_set]
    
    # Ensure proper types
    result = []
    for left, right in best_bicliques:
        left_clean = [int(x) for x in left]
        right_clean = [int(x) for x in right]
        result.append([left_clean, right_clean])
    
    return result
# EVOLVE-BLOCK-END
