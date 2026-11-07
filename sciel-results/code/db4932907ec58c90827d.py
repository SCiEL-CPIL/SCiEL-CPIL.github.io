import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Map edges to sets for fast lookup
    # adj[u] is a set of neighbors
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Function to find a maximal biclique containing a given edge (u, v)
    # using only edges from the 'available_edges' set
    def find_maximal_biclique(u, v, available_edges):
        L = {u}
        R = {v}
        
        # Helper to check if a vertex can be added to L
        # w can be added to L if (w, r) is in available_edges for all r in R
        def can_add_to_L(w):
            if w in L or w in R:
                return False
            for r in R:
                if (w, r) not in available_edges:
                    return False
            return True
            
        def can_add_to_R(z):
            if z in L or z in R:
                return False
            for l in L:
                if (l, z) not in available_edges:
                    return False
            return True

        changed = True
        while changed:
            changed = False
            # Try to add to L
            new_L = set()
            for w in range(n):
                if can_add_to_L(w):
                    new_L.add(w)
            if new_L:
                L = L.union(new_L)
                changed = True
            
            # Try to add to R
            new_R = set()
            for z in range(n):
                if can_add_to_R(z):
                    new_R.add(z)
            if new_R:
                R = R.union(new_R)
                changed = True
                
        return L, R

    # Greedy partition algorithm
    def greedy_partition(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # List of edges as tuples
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges:
            # Pick an edge to start a biclique
            # Strategy: Pick an edge with high degree in remaining graph? 
            # Or just pick one. Randomization helps.
            # Let's pick a random edge from remaining_edges
            # But to be deterministic in behavior if seed is fixed, we use random
            e = random.choice(list(remaining_edges))
            u, v = e
            
            # Find maximal biclique containing e
            L, R = find_maximal_biclique(u, v, remaining_edges)
            
            # Collect edges of this biclique
            biclique_edges = set()
            for l in L:
                for r in R:
                    if (l, r) in remaining_edges:
                        biclique_edges.add((l, r))
                    elif (r, l) in remaining_edges:
                        biclique_edges.add((r, l))
                        
            # Remove these edges from remaining
            remaining_edges.difference_update(biclique_edges)
            
            # Store biclique
            # Ensure lists are sorted for consistency? Not required but good.
            partition.append([sorted(L), sorted(R)])
            
        return partition

    # Local search: Merge bicliques if possible
    def local_search_merge(partition):
        # Check if two bicliques can be merged
        # B1=(L1, R1), B2=(L2, R2)
        # Union is (L1 U L2, R1 U R2)
        # Valid if all edges in (L1 U L2) x (R1 U R2) are in the original graph
        # AND the union of edges of B1 and B2 is exactly the set of edges in the union biclique
        # Wait, if B1 and B2 are edge-disjoint, their union edge set is E1 U E2.
        # The merged biclique covers (L1 U L2) x (R1 U R2).
        # This set must be a subset of the graph edges.
        # Also, it must be exactly E1 U E2?
        # No, the partition covers the graph edges.
        # If we merge, the new biclique covers more edges?
        # If (L1 U L2) x (R1 U R2) contains edges NOT in E1 U E2, those edges must have been covered by OTHER bicliques.
        # So we can only merge if (L1 U L2) x (R1 U R2) == E1 U E2.
        # This is a strong condition.
        
        # Actually, simpler check:
        # Can we replace B1 and B2 with B_merged?
        # B_merged must be a biclique.
        # The edges covered by B_merged must be exactly the edges covered by B1 and B2.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(partition)):
                if changed: break
                for j in range(i+1, len(partition)):
                    L1, R1 = partition[i]
                    L2, R2 = partition[j]
                    L_u = set(L1).union(L2)
                    R_u = set(R1).union(R2)
                    
                    # Check if L_u x R_u is a biclique in the graph
                    is_biclique = True
                    for l in L_u:
                        for r in R_u:
                            if (l, r) not in edge_set and (r, l) not in edge_set:
                                is_biclique = False
                                break
                        if not is_biclique: break
                    
                    if not is_biclique:
                        continue
                        
                    # Check if the edges of the merged biclique are exactly E1 U E2
                    # E1 is edges between L1, R1
                    # E2 is edges between L2, R2
                    # E_merged is edges between L_u, R_u
                    
                    # Since B1 and B2 are from the partition, they are edge disjoint.
                    # We need to ensure no "extra" edges appear in E_merged that were covered by other bicliques.
                    # i.e., E_merged must be a subset of (E1 U E2).
                    # But E1 U E2 is a subset of E_merged (since L1 subset L_u etc).
                    # So we need E_merged == E1 U E2.
                    
                    # Count edges in E1
                    count_e1 = 0
                    for l in L1:
                        for r in R1:
                            if (l, r) in edge_set or (r, l) in edge_set:
                                count_e1 += 1
                    # Count edges in E2
                    count_e2 = 0
                    for l in L2:
                        for r in R2:
                            if (l, r) in edge_set or (r, l) in edge_set:
                                count_e2 += 1
                    # Count edges in E_merged
                    count_em = 0
                    for l in L_u:
                        for r in R_u:
                            if (l, r) in edge_set or (r, l) in edge_set:
                                count_em += 1
                            
                    if count_em == count_e1 + count_e2:
                        # Merge
                        partition[i] = [sorted(L_u), sorted(R_u)]
                        partition.pop(j)
                        changed = True
                        break
                        
        return partition

    best_partition = None
    best_count = float('inf')
    
    # Run multiple restarts
    # Time limit check
    start_time = time.time()
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() - start_time > 50: # Safety margin
            break
            
        part = greedy_partition(random_seed=i)
        part = local_search_merge(part)
        
        if len(part) < best_count:
            best_count = len(part)
            best_partition = part
            
    # Fallback if no partition found (should not happen)
    if best_partition is None:
        best_partition = greedy_partition(random_seed=0)
        
    # Convert to required format: list of [left_vertices, right_vertices]
    # Ensure types are Python int and lists
    result = []
    for left, right in best_partition:
        result.append([int(v) for v in left], [int(v) for v in right])
        
    return result
# EVOLVE-BLOCK-END
