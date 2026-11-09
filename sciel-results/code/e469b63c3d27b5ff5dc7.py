import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s limit

    n = 16
    total_edges = n * (n - 1) // 2
    
    # 1. Precompute edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            edge_list.append((u, v))
            idx += 1

    # 2. Precompute Q3 conflict graph
    # A conflict exists between two edges e1 and e2 if there is a set of 6 other edges
    # such that e1, e2 and those 6 form a Q3.
    # Equivalently, for each Q3 subgraph H (8 vertices, 12 edges), all pairs of edges in H
    # are in conflict.
    
    # We can find all Q3 subgraphs by iterating over 8-vertex subsets and checking if they contain a Q3.
    # However, that's C(16,8) = 12870 subsets. For each, we need to check if it contains a Q3.
    # A faster way: iterate over all 8-vertex subsets. For each, check if it contains a Q3.
    # If it does, add conflicts for all pairs of edges within that Q3.
    # Note: A set of 8 vertices might contain multiple Q3s. We just need to mark conflicts for at least one.
    # Actually, we need to mark conflicts for ALL edges that are part of ANY Q3 in the induced subgraph?
    # No. The conflict graph for MIS should have an edge between e1 and e2 if e1 and e2 CANNOT coexist in a Q3-free graph.
    # e1 and e2 cannot coexist if there exists a set of 6 other edges such that {e1, e2} U S is a Q3.
    # This is equivalent to: e1 and e2 are in conflict if they are both part of some Q3 in K16.
    # So we just need to find all Q3 subgraphs of K16 and add all pairs of edges within each Q3 to the conflict graph.
    
    # Optimization: We can iterate over all 8-vertex subsets.
    # For a subset S, it contains a Q3 if the induced subgraph G[S] contains a Q3.
    # Since K16 is complete, G[S] is K8. Does K8 contain a Q3? Yes.
    # Wait, the problem is to find a Q3-FREE subgraph of K16.
    # The conflict is: if we pick edges e1 and e2, do they force a Q3?
    # No, they don't force it immediately. They force it if we ALSO pick the other 10 edges of the Q3.
    # So the conflict graph should connect e1 and e2 if there is a Q3 Q such that e1, e2 in E(Q).
    # If we select a set of edges E' that is an independent set in this conflict graph, then no Q3 is fully contained in E'.
    # Why? If a Q3 Q were fully contained in E', then any pair of edges in Q would be in E', which means they are connected in the conflict graph, so E' would not be independent.
    # Conversely, if E' is independent, no Q3 is fully contained in E'.
    
    # So the strategy is:
    # 1. Build conflict graph H where nodes are edges of K16.
    # 2. Edge (e1, e2) in H if there exists a Q3 Q such that e1, e2 in E(Q).
    # 3. Find Maximum Independent Set in H.
    
    # How to build H efficiently?
    # Iterate over all 8-vertex subsets S of {0..15}.
    # For each S, check if it contains a Q3.
    # Actually, since K16 is complete, EVERY 8-vertex subset contains MANY Q3s.
    # Specifically, the number of Q3 subgraphs in K8 is quite large.
    # So we can just iterate over all 8-vertex subsets, find ONE Q3 in it, and add conflicts for all pairs of edges in that Q3.
    # Is that sufficient?
    # If e1 and e2 are in conflict, there exists a Q3 Q containing them. Q is contained in some 8-vertex set S.
    # When we process S, we find a Q3 Q' in S. It might not be Q.
    # If we only find ONE Q3 per S, we might miss conflicts.
    # We need to add conflicts for ALL pairs of edges that are together in ANY Q3.
    
    # Better approach:
    # Iterate over all 8-vertex subsets S.
    # For each S, find ALL Q3 subgraphs in S.
    # For each Q3 found, add all pairs of its edges to the conflict set.
    # This is expensive. C(16,8) = 12870. For each, finding all Q3s in K8.
    # Number of Q3s in K8:
    # A Q3 is a bipartite graph K_{4,4} minus a perfect matching.
    # To form a Q3 in K8, we partition 8 vertices into two sets of 4, A and B.
    # There are 35 such partitions (as in the seed code).
    # For each partition, the Q3 is determined.
    # So there are 35 Q3s in K8?
    # Wait, the seed code says "35 bipartitions".
    # Let's verify. Number of ways to split 8 items into two groups of 4 is C(8,4)/2 = 35.
    # For each split, the edges of the Q3 are all edges between A and B EXCEPT a perfect matching.
    # But the perfect matching is NOT unique?
    # No, a Q3 is a specific graph. It is the 3-dimensional hypercube.
    # The hypercube Q3 is bipartite. It has 8 vertices and 12 edges.
    # In K8, how many subgraphs isomorphic to Q3 are there?
    # A Q3 is defined by its vertex set and its edge set.
    # The edge set of a Q3 on a fixed vertex set is NOT unique?
    # Actually, Q3 is the only 3-regular bipartite graph on 8 vertices?
    # No. There are other 3-regular bipartite graphs on 8 vertices.
    # For example, K_{4,4} is 4-regular.
    # Q3 is 3-regular.
    # How many 3-regular bipartite graphs on 8 vertices?
    # The number of Q3s in K8 is the number of ways to embed Q3 in K8.
    # Aut(Q3) has size 48.
    # Number of labelings of Q3 is 8! / 48 = 40320 / 48 = 840.
    # So there are 840 subgraphs of K8 isomorphic to Q3.
    
    # So for each 8-vertex subset, there are 840 Q3s.
    # 12870 * 840 = ~10 million Q3s.
    # For each Q3, we add C(12, 2) = 66 conflicts.
    # Total conflicts ~ 660 million.
    # This is too many to store explicitly in a Python set of pairs.
    
    # Alternative:
    # We don't need to store ALL conflicts. We can check conflicts on the fly?
    # No, MIS needs the graph structure.
    
    # Maybe we can use a different conflict definition?
    # Or maybe we can reduce the search space?
    
    # Let's look at the "Idea target": "Formulate the problem as a Maximum Independent Set (MIS) on a conflict graph... solve this MIS using a fast greedy coloring heuristic".
    
    # If the conflict graph is too dense/large, MIS is hard.
    # But maybe we can approximate?
    
    # Let's try a different angle.
    # The seed code uses local search.
    # MIS is a global optimization.
    # If we can build the conflict graph, we can use a standard MIS solver (branch and bound).
    
    # Can we build the conflict graph faster?
    # Instead of iterating over all 8-vertex subsets, can we iterate over Q3s directly?
    # A Q3 is determined by its 8 vertices and its structure.
    # We can iterate over all 8-vertex subsets (12870).
    # For each, we need to find the 840 Q3s.
    # This is still heavy.
    
    # What if we just use a SAT solver or ILP?
    # "Python standard library, NumPy and SciPy are available."
    # No MILP solver.
    
    # Let's stick to MIS but optimize the conflict generation.
    # Maybe we don't need ALL conflicts.
    # If we just find a LARGE independent set, it's good enough.
    # A greedy MIS algorithm:
    # 1. Order vertices (edges of K16) randomly or by degree.
    # 2. Pick a vertex, add to independent set, remove it and its neighbors.
    # 3. Repeat.
    
    # This requires the adjacency list of the conflict graph.
    # Can we generate the adjacency list on the fly?
    # For a given edge e, what are its conflicts?
    # e is in conflict with e' if they are in a common Q3.
    # How many Q3s contain a specific edge e?
    # Total Q3s in K16:
    # Number of ways to choose 8 vertices: C(16, 8) = 12870.
    # Q3s per vertex set: 840.
    # Total Q3s = 12870 * 840 = 10,810,800.
    # Each Q3 has 12 edges.
    # Total edge-Q3 incidences = 10,810,800 * 12 = 129,729,600.
    # Total edges in K16 = 120.
    # Average Q3s per edge = 129,729,600 / 120 = 1,081,080.
    # So each edge is in ~1 million Q3s.
    # Each Q3 has 11 other edges.
    # So each edge is in conflict with ~1 million * 11 edges?
    # But there are only 119 other edges.
    # So the conflict graph is DENSE.
    # In fact, it's likely COMPLETE or close to it.
    # If the conflict graph is complete, the MIS is just 1 edge.
    # But we know we can have 42 edges.
    # So the conflict graph is NOT complete.
    # Why?
    # My conflict definition: e1 and e2 are in conflict if they are in SOME Q3.
    # If e1 and e2 are in conflict, we cannot pick BOTH.
    # If the graph is dense, we can't pick many.
    # But wait. If e1 and e2 are in a Q3, does that mean they CANNOT coexist in a Q3-free graph?
    # NO.
    # A Q3-free graph just cannot contain ALL 12 edges of a Q3.
    # It CAN contain 11 edges of a Q3.
    # So e1 and e2 CAN coexist, as long as we don't pick the other 10.
    # My MIS formulation is WRONG.
    # MIS on the "co-occurrence in Q3" graph finds a set of edges such that NO TWO are in a common Q3.
    # This is MUCH stronger than Q3-free.
    # Q3-free means NO Q3 is fully present.
    # My MIS formulation means NO TWO edges are in a common Q3.
    # This would yield a very small set.
    
    # So the MIS strategy as described in the prompt ("nodes represent potential edges... edges connect nodes that would complete a Q3 subgraph if both were present") is slightly ambiguous.
    # "would complete a Q3 subgraph if both were present" usually implies that the presence of both FORCES a Q3? No, that's not how graph containment works.
    # Maybe it means: e1 and e2 are in conflict if there is a set of other edges S such that {e1, e2} U S is a Q3.
    # This is the same as "e1 and e2 are in a common Q3".
    # So the prompt's MIS formulation leads to the "No two edges in a common Q3" constraint, which is too strong.
    
    # Unless... the prompt implies a different conflict graph?
    # "edges connect nodes that would complete a Q3 subgraph"
    # Maybe it means: e1 and e2 are in conflict if adding e2 to a graph containing e1 creates a Q3?
    # No, that depends on the rest of the graph.
    
    # Let's re-read carefully: "Formulate the problem as a Maximum Independent Set (MIS) on a conflict graph where nodes represent potential edges of K16 and edges connect nodes that would complete a Q3 subgraph if both were present"
    
    # This phrasing is tricky. "would complete... if both were present".
    # If I have e1 and e2, do they complete a Q3? No, they are just 2 edges.
    # Maybe it means: e1 and e2 are in conflict if they are part of a "minimal" Q3 extension?
    
    # Actually, there is a known reduction from "Max k-Set Packing" or similar to MIS.
    # But for subgraph avoidance, the standard formulation is:
    # Variables: x_e for each edge.
    # Constraints: For each Q3 Q, sum_{e in Q} x_e <= 11.
    # Maximize sum x_e.
    # This is an Integer Linear Program.
    
    # Can we solve this ILP with a custom branch and bound?
    # Yes.
    # This is a 0-1 ILP with 120 variables and ~10 million constraints.
    # But the constraints are structured.
    
    # Let's try a different approach: **Simulated Annealing** or **Genetic Algorithm**?
    # The seed uses local search (greedy + random restarts/edge swaps).
    # Simulated Annealing allows "worse" moves to escape local optima.
    # This is a different optimization method.
    
    # Strategy: Simulated Annealing.
    # State: A set of edges (bitset or set).
    # Move: Add an edge, remove an edge, or swap an edge.
    # Objective: Maximize number of edges, subject to Q3-free.
    # If a move creates a Q3, it is invalid (or penalized heavily).
    # Temperature: Start high, cool down.
    
    # This is materially different from the greedy local search in the seed.
    # The seed only accepts improvements. SA accepts worse states (if valid) or penalizes invalid states.
    # Actually, if we penalize invalid states, we can explore the space of ALL graphs, not just Q3-free ones.
    # Objective: m - penalty * (number of Q3s).
    # This allows escaping local optima in the Q3-free landscape.
    
    # Let's implement Simulated Annealing.
    
    # 1. Represent graph as a set of edges or a bitset.
    # 2. Function to count Q3s (or just check if Q3-free).
    #    Counting Q3s is expensive.
    #    Checking if Q3-free is also expensive (C(16,8) * 35).
    #    But we can optimize the check.
    
    # 3. Move:
    #    - Pick a random edge, flip it.
    #    - Calculate delta in objective.
    #    - If valid (Q3-free), accept if improves or with prob exp(-delta/T).
    #    - If invalid, accept with prob exp(-large_penalty/T).
    
    # To speed up Q3 check:
    # We can maintain a count of Q3s?
    # When we add an edge (u,v), only Q3s containing (u,v) can change.
    # We can precompute for each edge e, the list of Q3s containing e.
    # There are ~1 million Q3s per edge. Too many to store.
    
    # Alternative:
    # Use the "8-vertex subset" check but optimized.
    # The seed code checks C(14,6) = 3003 subsets for each edge addition.
    # 3003 * 35 = 105k ops.
    # In SA, we might do 100k moves. 10^10 ops. Too slow.
    
    # We need a faster Q3 check.
    # Or a different strategy.
    
    # Let's go back to the MIS idea but fix the formulation.
    # The ILP formulation is correct.
    # Can we solve the ILP using a greedy heuristic + local search on the ILP?
    
    # Actually, let's try **Genetic Algorithm**.
    # Population of graphs.
    # Fitness: Number of edges - Penalty * Q3_count.
    # Crossover: Union/Intersection of edges.
    # Mutation: Flip random edges.
    # Selection: Tournament.
    
    # This is a different design family (Evolutionary).
    
    # But Q3 counting is the bottleneck.
    
    # Let's look at the "Idea target" again.
    # "Formulate the problem as a Maximum Independent Set (MIS) on a conflict graph... solve this MIS using a fast greedy coloring heuristic"
    
    # If the prompt insists on MIS, maybe I should try to make it work.
    # Is there a conflict graph where MIS gives the right answer?
    # Yes, if the nodes are NOT edges, but something else?
    # No, "nodes represent potential edges".
    
    # Maybe the conflict is: e1 and e2 are in conflict if they CANNOT be in the same Q3-free graph?
    # No, that's not a static property.
    
    # Let's assume the prompt's MIS description is a heuristic for "find a set of edges that are 'compatible'".
    # If I implement a greedy MIS on the "co-occurrence in Q3" graph, I will get a very small set (clique number of the complement?).
    # Let's test the size.
    # If the conflict graph is dense, MIS is small.
    # If I get 10 edges, it's bad.
    
    # What if I use the COMPLEMENT of the conflict graph?
    # Nodes: Edges.
    # Edge in complement if e1 and e2 are NOT in any common Q3.
    # Then a Clique in the complement is a set of edges where NO TWO are in a common Q3.
    # This is the same as MIS in the original conflict graph.
    # Still too strong.
    
    # Okay, I will implement **Simulated Annealing** with a **fast Q3 check**.
    # To make Q3 check fast:
    # Precompute for each 8-vertex subset whether it contains a Q3?
    # No, it depends on the edges present.
    
    # Let's use the seed's `contains_q3_in_subset_fast` but optimize it further.
    # Or use a bitset representation for the adjacency matrix.
    
    # Actually, let's try **Tabu Search**.
    # It's a local search with memory.
    # But SA is easier to implement correctly.
    
    # Let's try to improve the seed's local search with **Random Restarts** and **Edge Swaps**.
    # The seed only adds edges. It never removes an edge to add two others.
    # A swap (remove e1, add e2, add e3) can escape local optima.
    
    # Strategy: **Local Search with Swaps (2-opt style)**.
    # 1. Start with a dense seed.
    # 2. Loop:
    #    a. Try to add a random edge. If valid, accept.
    #    b. If stuck, try a swap: remove a random edge e, add two random edges e1, e2.
    #       Check if the new graph is Q3-free.
    #       If valid and edge count increases, accept.
    #       If valid and edge count same, accept with probability.
    #    c. If stuck for long, restart with a new random seed.
    
    # This is materially different from the pure greedy addition in the seed.
    
    # Implementation details:
    # - Use bitsets for adjacency for fast checks.
    # - `is_q3_free()` check.
    
    # Let's implement this.

    # Bitset representation:
    # adj_mask[v] is an integer where bit u is 1 if (u,v) is an edge.
    
    def bitset_contains_q3(adj_masks):
        # Check if the graph defined by adj_masks contains a Q3.
        # Iterate over all 8-vertex subsets.
        # For each, check if it contains a Q3.
        
        # Optimization:
        # A Q3 is 3-regular bipartite.
        # In any 8-vertex subset, if it contains a Q3, the induced subgraph must have at least 12 edges.
        # We can filter subsets by edge count.
        
        # Precompute all 8-vertex subsets?
        # There are 12870.
        # For each, check edge count.
        # If < 12, skip.
        # If >= 12, check for Q3.
        
        # To check for Q3 in a subset S:
        # Use the 35 splits.
        
        all_splits = []
        seen = set()
        for A in combinations(range(8), 4):
            B = tuple(i for i in range(8) if i not in A)
            key = frozenset([frozenset(A), frozenset(B)])
            if key not in seen:
                seen.add(key)
                mask_a = 0
                mask_b = 0
                for i in A:
                    mask_a |= (1 << i)
                for i in B:
                    mask_b |= (1 << i)
                all_splits.append((mask_a, mask_b, A, B))
        
        # We can precompute the list of 8-vertex subsets once.
        # But we need to check them against the current adj_masks.
        
        # This function will be called many times.
        # We need it to be fast.
        
        # Let's precompute the 8-vertex subsets and their local index mappings?
        # No, the vertices are 0..15.
        
        # We can precompute a list of all 8-vertex subsets as tuples of vertex IDs.
        subsets_8 = list(combinations(range(16), 8))
        
        for subset in subsets_8:
            # Count edges in induced subgraph
            edge_count = 0
            for i in range(8):
                vi = subset[i]
                mask = adj_masks[vi]
                # Count bits in mask that correspond to other vertices in subset
                # Create a mask for the subset
                sub_mask = 0
                for j in range(8):
                    if i != j:
                        sub_mask |= (1 << subset[j])
                # Count bits in (mask & sub_mask)
                edge_count += bin(mask & sub_mask).count('1')
            
            # Each edge counted twice
            edge_count //= 2
            
            if edge_count < 12:
                continue
            
            # Check for Q3
            local_adj = [0] * 8
            for i in range(8):
                vi = subset[i]
                mask = 0
                for j in range(8):
                    if i != j and (adj_masks[vi] & (1 << subset[j])):
                        mask |= (1 << j)
                local_adj[i] = mask
            
            for mask_a, mask_b, A_idx, B_idx in all_splits:
                covered = 0
                is_matching = True
                missing_count = 0
                
                for i in A_idx:
                    for j in B_idx:
                        if not (local_adj[i] & (1 << j)):
                            missing_count += 1
                            if missing_count > 4:
                                is_matching = False
                                break
                            if (covered & (1 << i)) or (covered & (1 << j)):
                                is_matching = False
                                break
                            covered |= (1 << i)
                            covered |= (1 << j)
                    if not is_matching:
                        break
                if is_matching:
                    return True
        
        return False

    # This is still slow. 12870 subsets * 35 splits * ...
    
    # Let's use the seed's approach for the check, but optimize the main loop.
    # The seed's check was `contains_q3_in_subset_fast` which checked C(14,6) subsets for a specific edge addition.
    # For a full graph check, we need to check ALL 8-vertex subsets.
    
    # Maybe we can use a **Constraint Programming** approach?
    # No, let's stick to the Local Search with Swaps.
    # We can reuse the seed's `contains_q3_in_subset_fast` logic but adapt it.
    
    # Actually, the seed's logic for checking a new edge (u,v) was:
    # Check all 8-vertex subsets containing u and v.
    # This is 3003 subsets.
    # If we do a swap (remove e1, add e2, e3), we need to check if the resulting graph is Q3-free.
    # This requires checking ALL 8-vertex subsets?
    # No. Only subsets that are affected by the change.
    # A Q3 is affected if it contains at least one of the changed edges.
    # Changed edges: e1 (removed), e2, e3 (added).
    # So we only need to check 8-vertex subsets that contain at least one of e1, e2, e3.
    # Number of such subsets:
    # Total subsets: 12870.
    # Subsets NOT containing e1, e2, e3:
    # This is complex to count directly.
    # But we can just iterate over all 12870 subsets and check?
    # 12870 * 35 = 450k ops per swap check.
    # If we do 1000 swaps, 450M ops. Might be tight.
    
    # Let's try to implement the Local Search with Swaps.

    # Initialize
    adj = [[0] * n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v or adj[u][v]: return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))
        
    def remove_edge(u, v):
        if not adj[u][v]: return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))
        
    # Seed: K3,13 + K3
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    best_edges = len(edges)
    best_edge_set = set(edges)
    
    # Precompute splits for Q3 check
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))
            
    def is_q3_free():
        # Check all 8-vertex subsets
        # Optimization: Use bitsets
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            adj_masks[u] = mask
            
        # Precomputed subsets
        for subset in subsets_8:
            # Fast edge count
            ec = 0
            for i in range(8):
                vi = subset[i]
                sub_mask = 0
                for j in range(8):
                    if i != j:
                        sub_mask |= (1 << subset[j])
                ec += bin(adj_masks[vi] & sub_mask).count('1')
            ec //= 2
            if ec < 12:
                continue
                
            local_adj = [0] * 8
            for i in range(8):
                vi = subset[i]
                mask = 0
                for j in range(8):
                    if i != j and (adj_masks[vi] & (1 << subset[j])):
                        mask |= (1 << j)
                local_adj[i] = mask
                
            for mask_a, mask_b, A_idx, B_idx in all_splits:
                covered = 0
                is_matching = True
                missing_count = 0
                for i in A_idx:
                    for j in B_idx:
                        if not (local_adj[i] & (1 << j)):
                            missing_count += 1
                            if missing_count > 4:
                                is_matching = False
                                break
                            if (covered & (1 << i)) or (covered & (1 << j)):
                                is_matching = False
                                break
                            covered |= (1 << i)
                            covered |= (1 << j)
                    if not is_matching:
                        break
                if is_matching:
                    return False
        return True
        
    # Precompute subsets_8
    subsets_8 = list(combinations(range(16), 8))
    
    # Local Search with Swaps
    import random
    iterations = 0
    max_iterations = 20000
    last_improvement = start_time
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # 1. Try to add a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not adj[u][v]:
            add_edge(u, v)
            if is_q3_free():
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_edge_set = set(edges)
                    last_improvement = time.time()
            else:
                remove_edge(u, v)
                
        # 2. Try a swap
        # Pick a random existing edge to remove
        if edges:
            e1 = random.choice(list(edges))
            u1, v1 = e1
            remove_edge(u1, v1)
            
            # Pick two random edges to add
            added = []
            for _ in range(2):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 != v2 and not adj[u2][v2]:
                    add_edge(u2, v2)
                    added.append((u2, v2))
                    
            if is_q3_free():
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_edge_set = set(edges)
                    last_improvement = time.time()
            else:
                # Revert
                for u2, v2 in added:
                    remove_edge(u2, v2)
                add_edge(u1, v1)
                
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
