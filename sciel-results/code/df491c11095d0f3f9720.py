import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build adjacency using bitmasks for speed
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    def edges_to_biclique(edge_subset):
        """
        Check if edge_subset forms a complete bipartite graph (biclique).
        Returns [L, R] if valid, else None.
        """
        if not edge_subset:
            return None
        
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)
        
        # Bipartite check using BFS on the subgraph defined by edge_subset
        color = {}
        for v in vertices:
            if v not in color:
                stack = [v]
                color[v] = 0
                while stack:
                    curr = stack.pop()
                    # Find neighbors in the subgraph
                    for nb in range(n):
                        if nb not in vertices:
                            continue
                        if (min(curr, nb), max(curr, nb)) in edge_subset:
                            if nb not in color:
                                color[nb] = 1 - color[curr]
                                stack.append(nb)
                            elif color[nb] == color[curr]:
                                return None  # Not bipartite
        
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # Check completeness: all pairs in L x R must be in edge_subset
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in edge_subset:
                    return None
        
        return [L, R]
    
    def greedy_init():
        """
        Greedy set cover: repeatedly find the biclique covering the most remaining edges.
        """
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            best_bic = None
            best_count = 0
            
            # Sample edges to find candidate bicliques
            sample_edges = list(remaining)
            random.shuffle(sample_edges)
            
            for e in sample_edges[:15]:
                u, v = e
                L = {u}
                R = {v}
                
                # Expand L and R iteratively
                changed = True
                while changed:
                    changed = False
                    # Try adding to L
                    for x in range(n):
                        if x not in L and x not in R:
                            valid = True
                            for r in R:
                                if (min(x, r), max(x, r)) not in remaining:
                                    valid = False
                                    break
                            if valid:
                                L.add(x)
                                changed = True
                    # Try adding to R
                    for y in range(n):
                        if y not in L and y not in R:
                            valid = True
                            for l in L:
                                if (min(l, y), max(l, y)) not in remaining:
                                    valid = False
                                    break
                            if valid:
                                R.add(y)
                                changed = True
                
                # Count edges covered
                count = 0
                for l in L:
                    for r in R:
                        if (min(l, r), max(l, r)) in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_bic = [list(L), list(R)]
            
            if best_bic is None:
                u, v = next(iter(remaining))
                best_bic = [[u], [v]]
            
            bicliques.append(best_bic)
            
            # Remove covered edges
            L, R = best_bic
            for l in L:
                for r in R:
                    remaining.discard((min(l, r), max(l, r)))
        
        return bicliques
    
    def try_merge(bicliques):
        """
        Try to merge two bicliques if their union is also a biclique.
        """
        n_bic = len(bicliques)
        for i in range(n_bic):
            for j in range(i + 1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Get union of edges
                edges_i = set()
                for u in L1:
                    for v in R1:
                        edges_i.add((min(u, v), max(u, v)))
                
                edges_j = set()
                for u in L2:
                    for v in R2:
                        edges_j.add((min(u, v), max(u, v)))
                
                union_edges = edges_i | edges_j
                
                # Check if union forms a biclique
                merged = edges_to_biclique(union_edges)
                if merged:
                    new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                    new_bicliques.append(merged)
                    return new_bicliques
        return None
    
    def try_split_and_replace(bicliques):
        """
        Try to split a biclique and replace it with two smaller ones that cover the same edges.
        This is less common but can help escape local optima.
        """
        # For now, skip this for simplicity
        return None
    
    # Main search loop
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        bicliques = greedy_init()
        
        # Local search: merge
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_bic = try_merge(bicliques)
            if new_bic is not None:
                bicliques = new_bic
                improved = True
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
            # If we found a very good solution, try to polish it
            if best_count <= 10:
                # Try more aggressive local search
                pass
    
    # Fallback
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
