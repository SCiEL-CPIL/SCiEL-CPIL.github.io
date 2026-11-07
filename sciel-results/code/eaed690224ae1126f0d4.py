# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
sys.setrecursionlimit(10000)
import time

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    m = len(edge_set)
    if m == 0:
        return []
    
    # Build edge list with indices
    edge_list = list(edge_set)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    
    # Find all maximal bicliques
    def find_maximal_bicliques():
        """Find all maximal bicliques in the graph."""
        maximal_bicliques = []
        seen = set()
        
        for e in edge_list:
            u, v = tuple(e)
            
            L = {u}
            R = {v}
            
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[y] for y in R):
                            L.add(x)
                            changed = True
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[y] for y in L):
                            R.add(x)
                            changed = True
            
            key1 = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))
            
            if key1 not in seen and key2 not in seen:
                seen.add(key1)
                seen.add(key2)
                maximal_bicliques.append((L, R))
        
        return maximal_bicliques
    
    max_bicliques = find_maximal_bicliques()
    num_bicliques = len(max_bicliques)
    
    # For each biclique, compute the set of edges it covers
    biclique_edges = []
    for L, R in max_bicliques:
        e_set = set()
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in edge_index:
                    e_set.add(edge_index[e])
        biclique_edges.append(frozenset(e_set))
    
    # For each edge, find which bicliques contain it
    edge_to_bicliques = [[] for _ in range(m)]
    for bi in range(num_bicliques):
        for e in biclique_edges[bi]:
            edge_to_bicliques[e].append(bi)
    
    # Solve exact cover using backtracking
    best_count = m + 1
    best_solution = None
    memo = {}
    time_limit = 50  # seconds
    
    def solve(covered, current_count, current_solution):
        nonlocal best_count, best_solution
        
        if time.time() - start_time > time_limit:
            return
        
        if current_count >= best_count:
            return
        
        if len(covered) == m:
            best_count = current_count
            best_solution = list(current_solution)
            return
        
        state = covered
        if state in memo:
            if memo[state] <= current_count:
                return
        else:
            memo[state] = current_count
        
        # Find the uncovered edge with fewest candidate bicliques
        min_count = float('inf')
        min_edge = -1
        min_bicliques = None
        
        for i in range(m):
            if i not in covered:
                count = len(edge_to_bicliques[i])
                if count < min_count:
                    min_count = count
                    min_edge = i
                    min_bicliques = edge_to_bicliques[i]
        
        if min_edge == -1:
            return
        
        # Sort bicliques by size (larger first) for better pruning
        sorted_bis = sorted(min_bicliques, key=lambda bi: -len(biclique_edges[bi]))
        
        for bi in sorted_bis:
            new_edges = biclique_edges[bi]
            
            if new_edges & covered:
                continue
            
            new_covered = covered | new_edges
            solve(new_covered, current_count + 1, current_solution + [bi])
    
    solve(frozenset(), 0, [])
    
    if best_solution is None:
        # Fallback: greedy
        remaining = set(range(m))
        used_bicliques = set()
        cover = []
        while remaining:
            best_bi = -1
            best_count_edges = 0
            for bi in range(num_bicliques):
                if bi in used_bicliques:
                    continue
                new_edges = biclique_edges[bi]
                if new_edges & (set(range(m)) - remaining):
                    continue
                count = len(new_edges & remaining)
                if count > best_count_edges:
                    best_count_edges = count
                    best_bi = bi
            if best_bi == -1:
                break
            cover.append(best_bi)
            used_bicliques.add(best_bi)
            remaining -= biclique_edges[best_bi]
        
        best_solution = cover
    
    result = []
    for bi in best_solution:
        L, R = max_bicliques[bi]
        left = sorted(list(L))
        right = sorted(list(R))
        if left and right and not set(left) & set(right):
            result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
