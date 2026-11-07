import time
import random
import numpy as np

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges to (u, v) with u < v
    normalized_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        normalized_edges.add((u, v))
    
    # Adjacency matrix for fast checks
    adj = np.zeros((n, n), dtype=bool)
    for u, v in normalized_edges:
        adj[u, v] = True
        adj[v, u] = True
        
    def get_biclique_edges(L, R):
        """Get set of edges in biclique L x R"""
        # L and R are lists of ints
        # Return set of (min, max) tuples
        res = set()
        for u in L:
            for v in R:
                if u == v: continue
                if u > v:
                    u, v = v, u
                res.add((u, v))
        return res

    def grow_biclique(u, v, adj, n):
        """
        Grow a maximal biclique starting from edge (u, v).
        Returns (L, R) where L and R are lists of vertices.
        """
        L = {u}
        R = {v}
        # Vertices not in L or R
        candidates_L = set(range(n)) - L - R
        candidates_R = set(range(n)) - L - R
        
        changed = True
        while changed:
            changed = False
            # Expand L: x can join L if connected to all in R
            new_L = []
            for x in candidates_L:
                # Check if x is connected to all y in R
                # Using numpy for speed if R is large, but sets are fine for n=30
                if all(adj[x, y] for y in R):
                    new_L.append(x)
            if new_L:
                L.update(new_L)
                candidates_L.difference_update(new_L)
                changed = True
                
            # Expand R: y can join R if connected to all in L
            new_R = []
            for y in candidates_R:
                if all(adj[y, x] for x in L):
                    new_R.append(y)
            if new_R:
                R.update(new_R)
                candidates_R.difference_update(new_R)
                changed = True
                
        return list(L), list(R)

    def greedy_partition(uncovered_edges, adj, n, random_seed=None):
        """
        Greedy set cover using maximal bicliques.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        partition = []
        uncovered = set(uncovered_edges)
        
        while uncovered:
            best_biclique = None
            best_count = -1
            best_L = None
            best_R = None
            
            # Get a sample of edges to start growth from, to save time if too many
            # For n=30, |E| is small enough to iterate all
            edge_list = list(uncovered)
            # Shuffle to allow randomization
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                L, R = grow_biclique(u, v, adj, n)
                # Calculate how many edges from 'uncovered' are covered by this biclique
                biclique_edges = get_biclique_edges(L, R)
                count = len(biclique_edges & uncovered)
                
                if count > best_count:
                    best_count = count
                    best_biclique = biclique_edges
                    best_L = L
                    best_R = R
                    
            if best_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            partition.append((best_L, best_R))
            uncovered.difference_update(best_biclique)
            
        return partition

    def merge_bicliques(partition, adj, n):
        """
        Try to merge pairs of bicliques if possible.
        Returns new partition.
        """
        merged = True
        while merged:
            merged = False
            new_partition = []
            i = 0
            while i < len(partition):
                if i == len(partition) - 1:
                    new_partition.append(partition[i])
                    i += 1
                    continue
                
                L1, R1 = partition[i]
                L2, R2 = partition[i+1]
                
                # Check if L1 x R2 and L2 x R1 are all edges
                # And disjointness: (L1 U L2) cap (R1 U R2) is empty
                # Since L1 cap R1 is empty and L2 cap R2 is empty, we just need:
                # L1 cap R2 = empty, L2 cap R1 = empty
                # And edges exist for all cross pairs.
                
                can_merge = True
                if not (set(L1).isdisjoint(R2) and set(L2).isdisjoint(R1)):
                    can_merge = False
                
                if can_merge:
                    # Check edges L1 x R2
                    for u in L1:
                        for v in R2:
                            if not adj[u, v]:
                                can_merge = False
                                break
                        if not can_merge: break
                            
                if can_merge:
                    # Check edges L2 x R1
                    for u in L2:
                        for v in R1:
                            if not adj[u, v]:
                                can_merge = False
                                break
                        if not can_merge: break

                if can_merge:
                    # Merge
                    new_L = list(set(L1) | set(L2))
                    new_R = list(set(R1) | set(R2))
                    # Verify disjointness of new L and R
                    if set(new_L).isdisjoint(new_R):
                        new_partition.append((new_L, new_R))
                        i += 2
                        merged = True
                    else:
                        new_partition.append(partition[i])
                        i += 1
                else:
                    new_partition.append(partition[i])
                    i += 1
            partition = new_partition
        return partition

    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    best_partition = None
    best_score = float('inf')
    
    # Run multiple greedy passes with different random seeds
    # The first one is deterministic (seed 0)
    for seed in range(100):
        if time.time() > deadline:
            break
            
        part = greedy_partition(normalized_edges, adj, n, random_seed=seed)
        part = merge_bicliques(part, adj, n)
        
        # Validation and scoring
        # Score is just len(part)
        # But we must ensure it's a valid partition
        covered = set()
        valid = True
        for L, R in part:
            if not L or not R:
                valid = False
                break
            if not set(L).isdisjoint(R):
                valid = False
                break
            biclique_edges = get_biclique_edges(L, R)
            # Check if all edges in biclique are in original graph
            for e in biclique_edges:
                if e not in normalized_edges:
                    valid = False
                    break
            if not valid:
                break
            # Check if edges are already covered (partition property)
            if covered & biclique_edges:
                valid = False
                break
            covered.update(biclique_edges)
            
        if valid:
            if covered != normalized_edges:
                valid = False
            
        if valid:
            score = len(part)
            if score < best_score:
                best_score = score
                best_partition = part

    if best_partition is None:
        # Fallback to star decomposition if something went wrong
        # This should not happen
        fallback = []
        for u, v in normalized_edges:
            fallback.append(([u], [v]))
        return [[list(L), list(R)] for L, R in fallback]
        
    # Convert to required format: list of [left, right]
    # Ensure types are Python int and list
    result = []
    for L, R in best_partition:
        result.append([list(L), list(R)])
        
    return result
# EVOLVE-BLOCK-END
