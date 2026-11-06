import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    n = 16

    # Q3 edges in canonical labeling (vertices 0-7)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Precompute all 8-vertex subsets of 16 vertices as (mask, sorted_list)
    all_8subsets = []
    for subset in combinations(range(n), 8):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        all_8subsets.append((mask, subset))

    # For each 8-vertex subset, precompute the list of "required edge masks" 
    # that would indicate Q3 containment under some vertex mapping.
    # A Q3 exists in G[S] iff there exists a bijection f: {0..7} -> S 
    # such that for every (a,b) in q3_edges, edge (f(a), f(b)) exists in G.
    
    # Key optimization: For each 8-subset S, we can represent the "edge pattern" 
    # of G[S] as a 28-bit number (C(8,2)=28 possible edges within S).
    # Then we precompute all 28-bit patterns that contain Q3 as a subgraph.
    
    # Precompute: for each 8-subset, which 28-bit edge patterns contain Q3?
    # There are 2^28 = 268M possible patterns, too many to store explicitly.
    
    # Alternative: For each 8-subset S, precompute the set of all 
    # "Q3 edge patterns" (as 28-bit masks) that correspond to some embedding of Q3 into S.
    # Number of distinct embeddings: 8!/48 = 420 per subset.
    # So for each subset, we have up to 420 28-bit masks.
    
    # Better approach: Precompute a dictionary mapping from (subset_mask, edge_pattern) -> True
    # But that's 12870 * 420 = 5.4M entries, each with a 28-bit key.
    # Actually, let's precompute for each subset the list of 28-bit patterns that are Q3 patterns.
    
    # Even better: For each 8-subset S (as a sorted tuple), compute the set of all 
    # 28-bit edge patterns that contain a Q3. Store as a set of 28-bit ints.
    
    # But 12870 subsets * up to 420 patterns each = 5.4M ints. That's fine memory-wise.
    
    # Actually, let me think differently. The fastest check:
    # For a given graph adj (16 ints of 16 bits each), check if Q3 exists.
    # Enumerate all C(16,8) = 12870 subsets. For each subset S, compute the 28-bit 
    # edge pattern of G[S]. Check if this pattern "contains" any Q3 pattern.
    
    # To make this O(1) per subset, precompute for each subset S a lookup structure.
    # The 28-bit pattern has C(8,2) = 28 bits. We can't have a 2^28 table per subset.
    
    # Alternative fast check: Use the fact that Q3 is bipartite with parts of size 4.
    # Q3 vertices can be partitioned into two sets A={0,3,5,6} and B={1,2,4,7} 
    # (or similar). Actually, Q3 is bipartite: even-weight vertices vs odd-weight.
    # Vertices 0-7 in Q3: weight = popcount. 
    # Even: 0(0),3(2),5(2),6(2) -> wait, let me recheck.
    # In standard Q3 labeling: vertex i has binary representation of i.
    # 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Even popcount: 0(0),3(2),5(2),6(2) -> {0,3,5,6}
    # Odd popcount: 1(1),2(1),4(1),7(3) -> {1,2,4,7}
    # Q3 is bipartite with these two parts.
    
    # For a Q3 to exist in G[S], there must be a partition of S into A' and B' (|A'|=|B'|=4)
    # and a bijection from {0,3,5,6} to A' and {1,2,4,7} to B' such that all 12 Q3 edges are present.
    
    # This is still complex. Let me use a different fast approach:
    # Precompute for each 8-subset S, the list of all 420 Q3 edge patterns (28-bit masks).
    # Then for a given graph, for each subset S, compute G[S]'s 28-bit pattern and check
    # if it's a superset of any of the 420 patterns.
    
    # "Superset" check: pattern & q3_pattern == q3_pattern for some q3_pattern.
    # We can precompute a bitset or use a trie. But 420 checks per subset * 12870 subsets 
    # = 5.4M checks, each O(1) with bitwise AND. That's feasible!
    
    # Let's precompute: for each 8-subset index, a list of 28-bit Q3 patterns.
    
    # First, for a given 8-subset S (sorted list of 8 vertices), compute all 420 Q3 patterns.
    # A Q3 pattern is determined by a bijection from {0..7} to S.
    # Two bijections give the same pattern if they differ by an automorphism of Q3.
    # Automorphism group of Q3 has size 48 (hyperoctahedral group B3).
    # So 8!/48 = 420 distinct patterns.
    
    # To generate the 420 patterns: enumerate all 8! = 40320 permutations of {0..7} assigned to S,
    # compute the 28-bit pattern, and deduplicate.
    
    # But 12870 subsets * 40320 permutations = 519M, too slow to precompute.
    
    # Smarter: The 28-bit pattern depends only on which pairs of S-vertices are connected.
    # For a permutation p: {0..7} -> S, the pattern has bit (i,j) set iff 
    # edge (p(i), p(j)) exists in Q3.
    # Wait, I have it backwards. The pattern represents edges in G[S].
    # For a given embedding f: Q3_vertices -> S, the required edges in G[S] are 
    # {(f(a), f(b)) : (a,b) in Q3_edges}.
    # As a 28-bit mask over the 28 pairs of S, this is a specific 28-bit number.
    
    # Let me just precompute using a different method:
    # For each 8-subset S, the 28 pairs of S are indexed 0..27.
    # For each of the 420 distinct Q3 embeddings, compute the 28-bit mask.
    # Store these masks in a list for that subset.
    
    # To avoid 519M operations, note that the 420 masks for subset S are the same 
    # "shape" regardless of S - they just use different pair indices.
    # Actually no, the pair indices depend on S.
    
    # Let me just do it: 12870 subsets, for each generate 40320 permutations, compute mask, dedupe.
    # 12870 * 40320 = 519M. Too slow in Python.
    
    # Alternative: Use the bipartite structure. Q3 is bipartite with parts A={0,3,5,6}, B={1,2,4,7}.
    # For a Q3 to exist in G[S], we need to choose 4 vertices for A' and 4 for B' from S,
    # and check if the bipartite graph between A' and B' contains the specific Q3 bipartite pattern.
    
    # C(8,4) = 70 ways to split S into two 4-sets. For each split, check if the 
    # 4x4 bipartite adjacency contains the Q3 bipartite pattern under some labeling.
    # The Q3 bipartite edges: A={0,3,5,6}, B={1,2,4,7}
    # Edges: (0,1),(0,2),(3,1),(3,2),(5,4),(5,7),(6,4),(6,7)
    # Wait, let me list them properly from q3_edges:
    # (0,1),(0,2),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7),(0,4)
    # Bipartite edges (A-B): 
    # 0-1, 0-2, 0-4 (0 in A, 1,2,4 in B)
    # 3-1, 3-2, 3-7 (3 in A, 1,2,7 in B)
    # 5-4, 5-7 (5 in A, 4,7 in B)
    # 6-4, 6-7 (6 in A, 4,7 in B)
    # Total: 3+3+2+2 = 10? But Q3 has 12 edges. 
    # I'm missing edges within A and within B. Q3 is NOT bipartite in the sense I thought.
    # Wait, Q3 IS bipartite. Let me recheck.
    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    # 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Edge (0,1): 000-001, popcount diff 1. OK.
    # Edge (0,2): 000-010, popcount diff 1. OK.
    # Edge (0,4): 000-100, popcount diff 1. OK.
    # Edge (1,3): 001-011, popcount diff 1. OK.
    # Edge (1,5): 001-101, popcount diff 1. OK.
    # Edge (2,3): 010-011, popcount diff 1. OK.
    # Edge (2,6): 010-110, popcount diff 1. OK.
    # Edge (3,7): 011-111, popcount diff 1. OK.
    # Edge (4,5): 100-101, popcount diff 1. OK.
    # Edge (4,6): 100-110, popcount diff 1. OK.
    # Edge (5,7): 101-111, popcount diff 1. OK.
    # Edge (6,7): 110-111, popcount diff 1. OK.
    # All edges connect even-popcount to odd-popcount vertices. So Q3 IS bipartite.
    # A (even popcount) = {0,3,5,6}, B (odd popcount) = {1,2,4,7}
    # Bipartite edges:
    # 0 (in A) connects to: 1, 2, 4 (in B)
    # 3 (in A) connects to: 1, 2, 7 (in B)
    # 5 (in A) connects to: 4, 7 (in B)
    # 6 (in A) connects to: 4, 7 (in B)
    # That's 3+3+2+2 = 10 edges. But Q3 has 12 edges!
    # I must be missing 2 edges. Let me recount.
    # Ah wait, I think I mislabeled. Let me recheck vertex 3.
    # 3 = 011, popcount 2, even. OK.
    # 3 connects to: 1(001), 2(010), 7(111). Yes, 3 edges.
    # Hmm, 10 != 12. Let me list all 12 edges again and check bipartiteness.
    # (0,1): 0 in A, 1 in B. OK.
    # (0,2): 0 in A, 2 in B. OK.
    # (0,4): 0 in A, 4 in B. OK.
    # (1,3): 1 in B, 3 in A. OK.
    # (1,5): 1 in B, 5 in A. OK.
    # (2,3): 2 in B, 3 in A. OK.
    # (2,6): 2 in B, 6 in A. OK.
    # (3,7): 3 in A, 7 in B. OK.
    # (4,5): 4 in B, 5 in A. OK.
    # (4,6): 4 in B, 6 in A. OK.
    # (5,7): 5 in A, 7 in B. OK.
    # (6,7): 6 in A, 7 in B. OK.
    # All 12 edges are between A and B. But I only counted 10 from the A side.
    # Let me recount from A side:
    # 0: neighbors 1,2,4 -> 3 edges
    # 3: neighbors 1,2,7 -> 3 edges
    # 5: neighbors 4,7 -> 2 edges... wait, 5 also connects to 1! (1,5) is an edge.
    # 5: neighbors 1,4,7 -> 3 edges
    # 6: neighbors 2,4,7 -> 3 edges... wait, 6 connects to 2? (2,6) yes.
    # 6: neighbors 2,4,7 -> 3 edges
    # Total: 3+3+3+3 = 12. I missed some earlier.
    # Corrected bipartite adjacency:
    # A={0,3,5,6}, B={1,2,4,7}
    # 0: {1,2,4}
    # 3: {1,2,7}
    # 5: {1,4,7}
    # 6: {2,4,7}
    
    # OK so for the fast check, for each 8-subset S, we try all C(8,4)=70 ways to pick A' from S.
    # For each choice of A' (and B'=S\A'), we need to check if there exist bijections 
    # f: {0,3,5,6} -> A' and g: {1,2,4,7} -> B' such that all 12 Q3 edges are present.
    # This is a bipartite matching / constraint satisfaction problem.
    # 4! * 4! = 576 assignments per split, 70 splits per subset, 12870 subsets.
    # 12870 * 70 * 576 = 519M. Still too slow for precomputation in Python.
    
    # Let me use a completely different fast approach:
    # Use the backtracking has_q3 but make it much faster with better pruning.
    # Or, use a precomputed table indexed by the 128-bit adjacency pattern.
    # 2^128 is too large.
    
    # Practical approach: Optimize the backtracking has_q3 significantly.
    # Key optimizations:
    # 1. Use arrays instead of dicts for assignment
    # 2. Precompute Q3 adjacency as bitmasks
    # 3. Use degree-based ordering
    # 4. Early termination
    
    # Let me implement a fast has_q3 with bitmasks:
    
    # Q3 adjacency as bitmasks (8 bits each)
    q3_adj = [0] * 8
    for u, v in q3_edges:
        q3_adj[u] |= (1 << v)
        q3_adj[v] |= (1 << u)
    
    # Q3 degree ordering: process vertices in order of degree (all degree 3 in Q3)
    # Use a fixed order: 0, 1, 2, 3, 4, 5, 6, 7
    
    def has_q3(adj):
        """Fast Q3 check using backtracking with bitmask operations."""
        # adj: list of 16 ints, bit j set in adj[i] means edge (i,j) exists.
        # Try to embed Q3 into the graph.
        
        # For each choice of graph vertex for Q3 vertex 0:
        for g0 in range(n):
            # assign[0] = g0, assign[1..7] = -1
            assign = [-1] * 8
            assign[0] = g0
            used_mask = (1 << g0)
            
            # Backtrack on Q3 vertices 1-7
            def bt(qv):
                if qv == 8:
                    return True
                
                # Get Q3 neighbors of qv that are already assigned
                # and Q3 neighbors not yet assigned
                qv_neighbors = q3_adj[qv]
                
                # For each candidate graph vertex:
                for gv in range(n):
                    if (used_mask >> gv) & 1:
                        continue
                    
                    # Check if gv is adjacent to all already-assigned Q3-neighbors of qv
                    ok = True
                    # For each Q3 neighbor nq of qv that is already assigned:
                    # gv must be adjacent to assign[nq]
                    for nq in range(8):
                        if (qv_neighbors >> nq) & 1:
                            if nq < qv:  # already assigned (we assign in order 0,1,2,...)
                                gn = assign[nq]
                                if not (adj[gv] >> gn) & 1:
                                    ok = False
                                    break
                            elif nq == qv:
                                pass
                            else:
                                # nq > qv, not yet assigned, skip for now
                                pass
                    
                    if not ok:
                        continue
                    
                    # Additional pruning: check that gv has enough available neighbors
                    # for the unassigned Q3-neighbors of qv
                    # Count unassigned Q3-neighbors of qv
                    unassigned_count = 0
                    for nq in range(8):
                        if (qv_neighbors >> nq) & 1 and nq > qv:
                            unassigned_count += 1
                    
                    # gv must have at least unassigned_count neighbors among unassigned graph vertices
                    # that are compatible... this is complex, skip for now.
                    
                    assign[qv] = gv
                    used_mask_new = used_mask | (1 << gv)
                    if bt(qv + 1):
                        return True
                    assign[qv] = -1
                
                return False
            
            if bt(1):
                return True
        
        return False
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 120

    best_edges_count = 0
    best_adj = None

    def greedy_add(adj, edge_order, deadline):
        """Add edges in given order, skipping those that create Q3."""
        adj = adj[:]
        count = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if has_q3(adj):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                count += 1
        return adj, count

    def local_search_add(adj, count, deadline):
        """Try to add more edges by trying all missing edges."""
        adj = adj[:]
        count = count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if has_q3(adj):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    count += 1
                    improved = True
        return adj, count

    def local_search_swap(adj, count, deadline):
        """Try edge swaps: remove one edge, add another, to increase edge count."""
        adj = adj[:]
        count = count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all pairs (remove e1, add e2) where e1 present, e2 absent
            for u1, v1 in all_edges:
                if time.time() > deadline:
                    break
                if not (adj[u1] >> v1) & 1:
                    continue
                for u2, v2 in all_edges:
                    if time.time() > deadline:
                        break
                    if (adj[u2] >> v2) & 1:
                        continue
                    # Remove (u1,v1), add (u2,v2)
                    adj[u1] ^= (1 << v1)
                    adj[v1] ^= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if not has_q3(adj):
                        count += 1  # net change: -1 + 1 = 0, but we might enable more adds
                        # Actually count stays same. But the structure changed.
                        improved = True
                        # After a swap, try to add more edges
                        adj, count = local_search_add(adj, count, deadline)
                        return adj, count
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
        return adj, count

    # Start with K3,13: vertices 0,1,2 connected to all of 3-15
    # Edges: (0,3)..(0,15), (1,3)..(1,15), (2,3)..(2,15) = 3*13 = 39 edges
    adj_k313 = [0] * n
    for i in range(3):
        for j in range(3, 16):
            adj_k313[i] |= (1 << j)
            adj_k313[j] |= (1 << i)
    
    count_k313 = sum(bin(adj_k313[i]).count('1') for i in range(n)) // 2
    if not has_q3(adj_k313):
        best_edges_count = count_k313
        best_adj = adj_k313[:]
    
    # Try multiple random greedy orders
    trials = 0
    while time.time() < deadline - 30:
        trials += 1
        random.seed(trials * 12345 + 67890)

        edge_order = all_edges[:]
        random.shuffle(edge_order)

        adj = [0] * n
        adj, count = greedy_add(adj, edge_order, deadline - 15)

        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]

        # Local search on best
        if best_adj is not None and time.time() < deadline - 5:
            adj_ls, count_ls = local_search_add(best_adj, best_edges_count, deadline - 2)
            if count_ls > best_edges_count:
                best_edges_count = count_ls
                best_adj = adj_ls

        # Try swap-based local search occasionally
        if best_adj is not None and trials % 2 == 0 and time.time() < deadline - 10:
            adj_sw, count_sw = local_search_swap(best_adj, best_edges_count, deadline - 5)
            if count_sw > best_edges_count:
                best_edges_count = count_sw
                best_adj = adj_sw

        if trials > 200:
            break

    # Final local search
    if best_adj is not None and time.time() < deadline - 1:
        adj_ls, count_ls = local_search_add(best_adj, best_edges_count, deadline - 0.5)
        if count_ls > best_edges_count:
            best_edges_count = count_ls
            best_adj = adj_ls

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

construct()
