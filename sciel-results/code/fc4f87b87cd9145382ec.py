import time
import random
from collections import defaultdict

# The graph data from the seed program
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    # 1. Generate a pool of candidate bicliques
    # We want "large" bicliques to cover many edges.
    # We use a randomized greedy expansion from every edge.
    candidates = []
    
    # To ensure diversity, we run multiple passes with different random seeds/orderings
    # but since n=30 is small, we can be exhaustive on starting edges and randomize expansion.
    
    # First, let's find maximal bicliques using a specific heuristic:
    # For each edge (u,v), try to expand it as much as possible.
    # To get diverse candidates, we can randomize the order of vertex addition.
    
    def expand_biclique(start_u, start_v, random_state=None):
        L = {start_u}
        R = {start_v}
        changed = True
        # Use a random seed for this specific expansion if provided
        if random_state is not None:
            rng = random.Random(random_state)
        else:
            rng = random
            
        while changed:
            changed = False
            # Find all vertices that can be added to L
            # A vertex x can be added to L if it is connected to all vertices in R
            # AND (x, y) is an edge for all y in R.
            # Note: We must ensure we don't add vertices that create non-edges within the biclique structure?
            # No, a biclique L x R requires all (u,v) for u in L, v in R to be edges.
            # It does NOT require L to be a clique or R to be a clique.
            # So x can be added to L if adj[x][y] is True for all y in R.
            
            candidates_L = []
            for x in range(n):
                if x in L:
                    continue
                if all(adj[x][y] for y in R):
                    candidates_L.append(x)
                    
            candidates_R = []
            for y in range(n):
                if y in R:
                    continue
                if all(adj[x][y] for x in L):
                    candidates_R.append(y)
            
            # Randomly pick one from L candidates and one from R candidates to add
            # This creates different maximal bicliques depending on the random choices
            if candidates_L:
                x = rng.choice(candidates_L)
                L.add(x)
                changed = True
            if candidates_R:
                y = rng.choice(candidates_R)
                R.add(y)
                changed = True
                
        return L, R

    # Generate candidates
    # We'll generate a few candidates for each edge to capture different maximal structures
    num_candidates_per_edge = 3
    
    for u, v in edges:
        for k in range(num_candidates_per_edge):
            L, R = expand_biclique(u, v, random_state=k)
            # Convert to sorted lists for hashing/deduplication
            L_list = tuple(sorted(L))
            R_list = tuple(sorted(R))
            candidates.append((L_list, R_list))
            
    # Deduplicate candidates
    unique_candidates = list(set(candidates))
    
    # Filter out small candidates? Maybe keep all, but prioritize large ones in selection.
    # Actually, for set cover, we just need the list.
    
    # 2. Solve the Set Cover problem
    # We have a set of edges E.
    # We have a set of candidates C = {C1, C2, ...} where each Ci covers a subset of E.
    # We want to find the minimum subset of C that covers all of E.
    # This is NP-hard, but we can use a greedy heuristic:
    # 1. Start with uncovered = E.
    # 2. While uncovered is not empty:
    #    a. Find the candidate Ci that covers the most edges in uncovered.
    #    b. Add Ci to the solution.
    #    c. Remove the edges covered by Ci from uncovered.
    
    # To make it faster, we can precompute the edge sets for each candidate.
    # But with ~180 edges and maybe ~100 candidates, it's fast enough to compute on the fly or precompute.
    
    # Precompute edge sets for candidates
    candidate_edge_sets = []
    for L, R in unique_candidates:
        e_set = set()
        for u in L:
            for v in R:
                e_set.add((min(u, v), max(u, v)))
        candidate_edge_sets.append(e_set)
        
    # Greedy Set Cover
    uncovered = set(edge_set)
    selected_candidates = []
    
    # We need to be careful with time.
    # Number of candidates might be around 100-200.
    # Number of edges is ~180.
    # Greedy step: For each candidate, count intersection with uncovered.
    # This is O(|Candidates| * |Edges|) per step.
    # Steps are at most |Edges| (usually much less, around 5-10 bicliques).
    # So ~10 * 200 * 180 = 360,000 operations. Very fast.
    
    while uncovered:
        if time.time() > deadline - 5:
            # Fallback to a simple valid partition if we run out of time
            # This should not happen, but just in case
            break
            
        best_candidate_idx = -1
        max_covered = 0
        
        for i, e_set in enumerate(candidate_edge_sets):
            # Count how many edges in e_set are in uncovered
            # intersection size
            covered_count = len(e_set & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_candidate_idx = i
                
        if best_candidate_idx == -1:
            # No candidate covers any uncovered edge? This shouldn't happen if candidates are valid bicliques from the graph.
            # It might happen if we missed some edges in candidate generation?
            # No, every edge is a candidate itself (size 1 biclique).
            # So this should not happen.
            break
            
        selected_candidates.append(best_candidate_idx)
        uncovered -= candidate_edge_sets[best_candidate_idx]
        
    # 3. Construct the final result
    # The selected candidates form a partition?
    # Wait, the problem requires a PARTITION.
    # "The cross-edge sets must partition all pinned graph edges exactly once."
    # Set Cover allows overlaps.
    # If we select overlapping bicliques, edges are covered multiple times.
    # We need to ensure that the selected bicliques are DISJOINT in their edge sets.
    
    # The standard Greedy Set Cover does NOT guarantee disjointness.
    # However, if we choose the *maximal* bicliques, do they tend to be disjoint? No.
    
    # We need a different approach for PARTITION.
    # A partition is a specific type of cover where sets are disjoint.
    
    # Let's rethink.
    # If we use the greedy merging approach from the seed, it maintains the partition property.
    # The Set Cover approach breaks the partition property.
    
    # Can we adapt Set Cover to enforce disjointness?
    # Yes, by modifying the selection process:
    # When we select a candidate, we remove its edges from `uncovered`.
    # But we must also ensure that future candidates do NOT cover any of these edges.
    # This is equivalent to saying: we are looking for a subset of candidates that are pairwise disjoint in their edge sets, and their union is E.
    
    # This is still hard.
    # However, we can use the following heuristic:
    # 1. Generate candidates.
    # 2. Sort candidates by size (descending).
    # 3. Iterate through candidates.
    # 4. If a candidate is disjoint from all already selected candidates, select it.
    # 5. Remove its edges from the pool of "available" edges?
    #    Actually, if we just select disjoint bicliques, we might not cover ALL edges.
    #    We need to cover ALL edges.
    
    # Let's try a different strategy:
    # **Greedy Disjoint Set Cover**:
    # 1. Start with all edges uncovered.
    # 2. Generate candidates.
    # 3. While uncovered is not empty:
    #    a. Find the candidate that covers the most uncovered edges.
    #    b. BUT, we must ensure that this candidate does NOT cover any edge that is already covered by a previously selected candidate?
    #       No, if we select a candidate, we "own" those edges.
    #       The constraint is that the final set of bicliques must be a partition.
    #       This means no edge can be in two bicliques.
    #       So, if we select Biclique B1, we remove its edges from the graph.
    #       Then we look for B2 in the remaining graph.
    #       This is exactly what the "Greedy Partition" in the seed program does!
    #       It picks an edge, expands to a maximal biclique in the *remaining* graph, removes those edges, and repeats.
    
    # So the seed program's `greedy_partition` is essentially a Greedy Disjoint Set Cover.
    # The seed program then tries to *merge* these.
    
    # My new strategy should be:
    # 1. Use a better initial partition than the simple greedy.
    #    The simple greedy picks the *first* edge and expands.
    #    A better greedy might pick the edge that is part of the *largest* possible biclique.
    #    Or, pick the edge with the highest "potential" (e.g., endpoints have high degree).
    
    # 2. Then, use a more powerful local search than just pairwise merging.
    #    Maybe "Split and Re-merge" or "Swap"?
    
    # Let's try improving the Initial Partition first.
    # Instead of picking the first edge, pick the edge (u,v) that maximizes the size of the biclique it can form.
    # This is expensive if done for every edge.
    # But n=30, edges=180.
    # For each edge, expand to maximal biclique. 180 * (expansion cost).
    # Expansion cost is small.
    
    # Let's implement this "Best-First" greedy partition.
    
    def get_biclique_edges(L, R):
        return set((min(u, v), max(u, v)) for u in L for v in R)
    
    def greedy_partition_best_first():
        rem = set(edge_set)
        bics = []
        while rem:
            if time.time() > deadline - 5:
                # Fallback to simple greedy if running out of time
                # (This is a safety net, shouldn't trigger)
                return None
                
            # Find the edge in `rem` that yields the largest biclique
            best_edge = None
            best_L = None
            best_R = None
            best_size = 0
            
            # To speed up, we can sample a subset of edges if rem is large?
            # No, 180 edges is small.
            for e in rem:
                u, v = e
                # Expand (u,v) in the graph induced by `rem`?
                # Wait, the biclique must consist of edges that are in `rem`.
                # So when expanding, we can only add vertices x to L if (x, y) is in `rem` for all y in R.
                
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    # Add to L
                    for x in range(n):
                        if x in L: continue
                        # x must be connected to all in R via edges in `rem`
                        # AND x must not be in R (disjoint sides)
                        if x in R: continue
                        if all((min(x, y), max(x, y)) in rem for y in R):
                            L.add(x)
                            changed = True
                    # Add to R
                    for y in range(n):
                        if y in R: continue
                        if y in L: continue
                        if all((min(x, y), max(x, y)) in rem for x in L):
                            R.add(y)
                            changed = True
                
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R
                    best_edge = e
                    
            if best_L is None:
                break # Should not happen
                
            bics.append([list(best_L), list(best_R)])
            # Remove these edges from rem
            b_edges = get_biclique_edges(best_L, best_R)
            rem -= b_edges
            
        return bics

    current_bicliques = greedy_partition_best_first()
    
    if current_bicliques is None:
        # Fallback to simple greedy
        rem = set(edge_set)
        current_bicliques = []
        while rem:
            u, v = next(iter(rem))
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if x in R: continue
                    if all((min(x, y), max(x, y)) in rem for y in R):
                        L.add(x)
                        changed = True
                for y in range(n):
                    if y in R: continue
                    if y in L: continue
                    if all((min(x, y), max(x, y)) in rem for x in L):
                        R.add(y)
                        changed = True
            current_bicliques.append([list(L), list(R)])
            rem -= get_biclique_edges(L, R)
            
    # 2. Local Search: Merging
    # This part is similar to the seed, but we start from a better partition.
    
    def can_merge(L1, R1, L2, R2):
        for u in L1:
            for v in R2:
                if not adj[u][v]:
                    return False
        for u in L2:
            for v in R1:
                if not adj[u][v]:
                    return False
        return True

    def merge(L1, R1, L2, R2):
        new_L = list(set(L1) | set(L2))
        new_R = list(set(R1) | set(R2))
        return new_L, new_R

    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        indices = list(range(len(current_bicliques)))
        random.shuffle(indices)
        
        for i_idx in range(len(indices)):
            i = indices[i_idx]
            if i >= len(current_bicliques): continue
            for j_idx in range(i_idx + 1, len(indices)):
                j = indices[j_idx]
                if j >= len(current_bicliques): continue
                
                L1, R1 = current_bicliques[i]
                L2, R2 = current_bicliques[j]
                
                if can_merge(L1, R1, L2, R2):
                    new_L, new_R = merge(L1, R1, L2, R2)
                    current_bicliques.pop(j)
                    current_bicliques[i] = [new_L, new_R]
                    improved = True
                    break
            if improved:
                break
                
    # 3. Final validation and type conversion
    covered_edges = set()
    for L, R in current_bicliques:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    return [] # Overlap found
                covered_edges.add(e)
                if not adj[u][v]:
                    return [] # Invalid edge
    
    if covered_edges != edge_set:
        return [] # Missing edges
        
    final_result = []
    for L, R in current_bicliques:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
