import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    random.seed(42)
    
    # Map global edges to indices 0..119
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # Local edge indices for K8 (28 edges)
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # Generate Q3 masks for K8
    # A Q3 is a bipartite graph K_{4,4} minus a perfect matching.
    # We enumerate all such subgraphs on 8 labeled vertices.
    seen_masks = set()
    q3_masks = []
    for A in itertools.combinations(range(8), 4):
        A_set = set(A)
        B = tuple(v for v in range(8) if v not in A_set)
        # A and B are the bipartition.
        # The edges present in Q3 are all (a,b) except (a, match(a))
        # We iterate over all perfect matchings between A and B.
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                matched = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != matched:
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks.append(mask)
                
    # Precompute all 8-subsets of vertices
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # For each subset, map local edge indices (0..27) to global edge indices (0..119)
    subset_l2g = []
    for s in all_subsets:
        l2g = []
        for i in range(8):
            for j in range(i+1, 8):
                u_glob, v_glob = s[i], s[j]
                g_idx = edge_index[u_glob][v_glob]
                l2g.append(g_idx)
        subset_l2g.append(l2g)
        
    # Build hypergraph of constraints
    # A hyperedge is a set of global edges that must NOT all be present.
    # Specifically, if a Q3 pattern is fully present in a subset, it's a violation.
    # A Q3 pattern has 12 edges.
    # We can represent a constraint as a frozenset of global edge indices.
    # However, checking if a set of edges contains a Q3 is equivalent to checking if it contains any of these hyperedges.
    # Actually, it's simpler: A graph is Q3-free iff it does NOT contain any of these hyperedges as a subgraph.
    # Wait, the hyperedge IS the set of edges of the Q3.
    # So if the set of chosen edges intersects a hyperedge in ALL 12 edges, it's invalid.
    # This is a 12-uniform hypergraph.
    
    # To use MIS, we need a conflict graph where edges represent "cannot be chosen together".
    # But here, the conflict is of arity 12.
    # We can reduce this to a standard MIS problem by introducing auxiliary nodes?
    # Or we can use a branch and bound that handles hyperedges directly.
    
    # Alternative: Since 12 is small, we can precompute for every pair of edges, the set of Q3s they are part of?
    # No, let's stick to the direct B&B on the hypergraph.
    
    # Hyperedges: list of sets of global edge indices.
    # Number of hyperedges: 140 subsets * 480 Q3 masks?
    # 480 is the number of distinct Q3 masks on 8 vertices.
    # Total hyperedges ~ 140 * 480 = 67200.
    # Each has 12 edges.
    
    hyperedges = set()
    for s_idx, l2g in enumerate(subset_l2g):
        for q_mask in q3_masks:
            # Extract the 12 local edges present in this Q3 mask
            local_edges_in_q3 = []
            for l in range(28):
                if (q_mask >> l) & 1:
                    local_edges_in_q3.append(l)
            # Map to global
            global_edges_in_q3 = tuple(sorted(l2g[l] for l in local_edges_in_q3))
            hyperedges.add(frozenset(global_edges_in_q3))
            
    hyperedges_list = list(hyperedges)
    
    # Optimization: If a hyperedge H contains another hyperedge H', H' is redundant?
    # No, they are all size 12.
    # But if two hyperedges share 11 edges, etc.
    
    # Let's use a Branch and Bound algorithm for Maximum Independent Set in a hypergraph.
    # Actually, it's easier to view this as: Find max subset of edges E' such that no hyperedge is a subset of E'.
    
    # We can use a recursive search.
    # State: current set of chosen edges (bitmask), current size, index of next edge to consider.
    # This is too slow for 120 edges if we just do standard B&B.
    
    # Better approach: Use the fact that we want to MAXIMIZE edges.
    # This is equivalent to MINIMIZING removed edges.
    # Let R be the set of removed edges.
    # Constraint: For every hyperedge H, H is NOT a subset of (AllEdges \ R).
    # i.e., H \ R is not empty. i.e., R intersects H.
    # So R must be a Hitting Set (Transversal) of the hypergraph.
    # We want to minimize |R|.
    # This is the Minimum Hitting Set problem, which is NP-hard.
    
    # Since the hyperedges are size 12, and we expect to remove ~10-20 edges (based on score 1.25 -> ~150 edges kept? No, 1.25 * seed. Seed is likely 100-110. 1.25*100=125? Max is 120.
    # Wait, score 1.2564. If seed is 100, score is 125? No, max edges is 120.
    # The task says "Maximize the number of edges".
    # If the seed program returns ~100 edges, and we return 125? Impossible.
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # If seed is 100, and we get 125, score is 1.25.
    # So we are looking for a hitting set of size 120 - 125 = -5? Impossible.
    # Maybe the seed value is lower.
    # Let's assume we want to keep as many edges as possible.
    
    # Algorithm: Branch and Bound for Minimum Hitting Set.
    # Nodes in recursion:
    # 1. Current hitting set R (bitmask or list).
    # 2. Remaining hyperedges to hit.
    # 3. Lower bound on size of R.
    
    # To make this fast, we need good pruning.
    # We can order edges by frequency in hyperedges.
    
    # Precompute: For each edge, which hyperedges it hits.
    edge_to_hyperedges = [[] for _ in range(total_edges)]
    for h_idx, h in enumerate(hyperedges_list):
        for e in h:
            edge_to_hyperedges[e].append(h_idx)
            
    # Sort edges by frequency (descending)
    edge_freq = [len(edge_to_hyperedges[e]) for e in range(total_edges)]
    sorted_edges = sorted(range(total_edges), key=lambda x: -edge_freq[x])
    
    # We will search for a hitting set.
    # We want to find the smallest hitting set.
    
    best_hitting_set_size = total_edges # Start with worst case (remove all)
    best_hitting_set = set(range(total_edges))
    
    # To speed up, we can use a greedy initialization to get a good upper bound.
    # Greedy Hitting Set:
    # While hyperedges remain:
    #   Pick edge that hits most remaining hyperedges.
    #   Add to R.
    #   Remove all hyperedges hit by this edge.
    
    remaining_hes = set(range(len(hyperedges_list)))
    greedy_R = set()
    
    # Precompute for each edge, the set of hyperedges it hits (as a set of indices for fast removal)
    # We already have edge_to_hyperedges as list of indices.
    
    # For efficiency in greedy, maintain count of how many times each edge appears in remaining hyperedges.
    # This is dynamic.
    
    # Simpler Greedy:
    # 1. Count freq of each edge in remaining hyperedges.
    # 2. Pick max freq edge.
    # 3. Remove all hyperedges containing that edge.
    # 4. Update counts.
    
    # Since 67k hyperedges is large, we need to be careful with time.
    # But 120 edges is small.
    
    # Let's implement a simple recursive B&B for Hitting Set.
    # It might be too slow if not pruned well.
    # Given the deadline, a robust heuristic + local search might be safer than exact B&B if B&B explodes.
    # However, "Minimum Hitting Set" on 12-uniform hypergraph with 120 vertices.
    # The expected size of the hitting set is small (~10-20).
    
    # Let's try a local search approach on the Hitting Set formulation.
    # Start with a greedy hitting set R.
    # Try to remove edges from R if the hitting property is still maintained.
    # If we can remove an edge, do it.
    # If stuck, try to swap an edge in R with an edge not in R to allow removing more?
    
    # Actually, the "repair" strategy in the current program is effectively finding a hitting set by removing edges from violations.
    # The new strategy:
    # 1. Greedy Hitting Set to get initial R.
    # 2. Local Search: 
    #    - Try to delete any edge from R. If valid, delete.
    #    - If no deletion possible, try to add an edge to R (bad move) to enable multiple deletions? No.
    #    - Standard local search for Min Hitting Set: 
    #       Neighborhood: Remove one edge from R. (Only valid if R remains a hitting set).
    #       This is a "downward" search. It gets stuck in local minima.
    #       To escape, we can use "k-exchange": Remove k edges, add 1 edge?
    
    # Let's use a different perspective:
    # We want to maximize kept edges.
    # Start with all edges kept.
    # Violations are hyperedges fully present.
    # This is the original formulation.
    
    # Let's go with **Simulated Annealing** on the edge set.
    # State: Bitmask of edges present.
    # Energy: Number of Q3 violations (hyperedges fully present).
    # We want Energy = 0.
    # Maximize edges subject to Energy = 0.
    
    # Objective: Maximize (EdgeCount - Penalty * Violations).
    # If Penalty is large, we prioritize removing violations.
    
    # Steps:
    # 1. Start with all edges present.
    # 2. While violations > 0:
    #      Pick a random violated hyperedge.
    #      Remove a random edge from it.
    # 3. Local Search:
    #      Try to add back removed edges.
    #      If adding edge e creates a violation, reject (or accept with prob based on temp).
    #      If adding edge e does not create a violation, accept.
    #      Also try swapping: Remove e1, Add e2.
    
    # This is similar to the current program but with SA dynamics to escape local optima.
    
    # Implementation details:
    # We need to quickly calculate the change in violation count when adding/removing an edge.
    
    # Precomputation:
    # For each edge e, list of hyperedges containing e.
    # For each hyperedge h, bitmask of edges in h.
    
    # State:
    # present: set of edges present (or bitmask)
    # For each hyperedge h, count of edges in h that are present.
    # If count == 12, it's a violation.
    
    # When we add edge e:
    #   For each h in edge_to_hyperedges[e]:
    #       count[h] += 1
    #       if count[h] == 12: violations += 1
    #
    # When we remove edge e:
    #   For each h in edge_to_hyperedges[e]:
    #       count[h] -= 1
    #       if count[h] == 11: violations -= 1
    
    # This is O(degree(e)) per step, which is fast.
    
    # Let's implement this.
    
    # 1. Init
    present = [1] * total_edges
    counts = [0] * len(hyperedges_list)
    violations = 0
    
    # Initialize counts
    for h_idx, h in enumerate(hyperedges_list):
        counts[h_idx] = len(h) # All edges present initially
        if counts[h_idx] == 12:
            violations += 1
            
    # 2. Phase 1: Eliminate violations greedily/randomly
    #    While violations > 0:
    #       Find a violated hyperedge.
    #       Remove a random edge from it.
    
    # To find a violated hyperedge quickly, we can keep a list of violated hyperedges?
    # Or just scan. 67k scan is fast.
    
    # Better: Maintain a set of violated hyperedges.
    violated_set = set(i for i, c in enumerate(counts) if c == 12)
    
    while violated_set and time.time() < deadline - 10.0:
        # Pick random violated hyperedge
        v_idx = random.choice(list(violated_set))
        h = hyperedges_list[v_idx]
        
        # Pick random edge from h that is present
        # All edges in h are present if it's violated?
        # Yes, because count == 12 means all 12 are present.
        e_to_remove = random.choice(list(h))
        
        # Remove e_to_remove
        present[e_to_remove] = 0
        
        # Update counts
        for h_idx2 in edge_to_hyperedges[e_to_remove]:
            counts[h_idx2] -= 1
            if counts[h_idx2] == 12:
                violated_set.add(h_idx2)
            elif counts[h_idx2] == 11:
                violated_set.discard(h_idx2)
                
    # 3. Phase 2: Local Search / Simulated Annealing to add back edges
    #    We want to maximize edges.
    #    Try adding random removed edges.
    #    If it doesn't increase violations, accept.
    #    If it increases violations, accept with probability exp(-delta / T).
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # List of removed edges
    removed_edges = [i for i in range(total_edges) if not present[i]]
    
    # To make it faster, we can shuffle removed edges and try to add them in random order.
    # But SA allows revisiting.
    
    # Let's do a simple randomized hill climbing first, then SA if time permits.
    # Actually, just do SA from the start of Phase 2.
    
    current_edges = sum(present)
    best_edges = current_edges
    best_present = present[:]
    
    # We need to track violations count for the SA step.
    # current_violations = len(violated_set)
    
    # To speed up, we can just try to add edges one by one.
    # If we can add an edge without creating a violation, we do it immediately (greedy).
    # This is a "forward repair".
    
    # Let's do a pass of greedy addition.
    # Shuffle removed edges
    random.shuffle(removed_edges)
    
    for e in removed_edges:
        if present[e]:
            continue
        # Check if adding e creates a violation
        # It creates a violation if any h in edge_to_hyperedges[e] has count 11.
        creates_violation = False
        for h_idx in edge_to_hyperedges[e]:
            if counts[h_idx] == 11:
                creates_violation = True
                break
                
        if not creates_violation:
            # Add e
            present[e] = 1
            for h_idx in edge_to_hyperedges[e]:
                counts[h_idx] += 1
                if counts[h_idx] == 12:
                    violated_set.add(h_idx)
            removed_edges.remove(e) # Or mark as present
            
    # Now try swaps: Remove e1, Add e2.
    # This is more complex.
    
    # Let's stick to the greedy addition which is very fast.
    # Then, if time, do a few random restarts or perturbations.
    
    # Convert to result
    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if present[e_idx]:
            result.append([u, v])
            
    return result

