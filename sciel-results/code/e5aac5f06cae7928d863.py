"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def clique_partition(edges, n, k):
        # Build adjacency structure
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # We need to cover all edges in edge_set with cliques.
        # Each clique must be a set of vertices where all pairs are in edge_set.
        # We want to minimize the number of cliques.
        
        # Strategy: Greedy Set Cover with "Maximal Clique" candidates.
        # 1. Generate a set of candidate cliques.
        #    - The full K clique (size k).
        #    - For each v in I, the clique formed by {v} U {u in K | (u,v) in edges}.
        #      This is a maximal clique containing v and only K vertices.
        #    - Also consider pairs (u, v) as fallback.
        #    - Actually, any subset of a clique is a clique. But we want large ones.
        #    - In split graphs, every maximal clique is either K itself, or {v} U N(v) for some v in I.
        #      Proof: If a clique contains two vertices from I, say v1, v2, they are not adjacent (I is independent).
        #      So a clique can contain at most one vertex from I.
        #      If it contains no vertex from I, it's a subset of K.
        #      If it contains exactly one vertex v from I, it must be a subset of {v} U N_K(v).
        #      Since we want to cover edges efficiently, larger cliques are better.
        #      So the "maximal" cliques to consider are:
        #      1. K (covers all edges within K)
        #      2. C_v = {v} U N_K(v) for each v in I (covers edges within N_K(v) and all cross edges incident to v)
        
        # Let's verify this logic.
        # If we pick C_v, it covers:
        # - All edges (u, w) where u, w in N_K(v).
        # - All edges (u, v) where u in N_K(v).
        # It does NOT cover edges (u, w) where u in N_K(v) and w in K \ N_K(v) (since w not in C_v).
        # It does NOT cover edges (u, v') for v' != v.
        
        # So we have a set of "maximal" cliques. Any clique in the graph is a subset of one of these.
        # It is never beneficial to use a non-maximal clique if we can extend it to a maximal one, 
        # because extending it covers more edges (or same edges if the added vertices have no new edges to cover, 
        # but adding vertices doesn't hurt validity). 
        # Wait, does adding a vertex w to a clique C make it still a clique? Yes, if w is adjacent to all in C.
        # If C is a clique and w is adjacent to all in C, then C U {w} is a clique.
        # Using C U {w} covers all edges covered by C, plus edges incident to w.
        # So we should always use maximal cliques.
        
        # Candidate Cliques:
        candidates = []
        
        # 1. The K clique
        if k >= 2:
            candidates.append(frozenset(range(k)))
            
        # 2. Cliques centered at each v in I
        # N_K(v) = {u in K | (u, v) in edge_set}
        for v in range(k, n):
            nbrs = adj[v]
            # Filter to K only
            nbrs_k = [u for u in nbrs if u < k]
            if len(nbrs_k) >= 1: # Need at least one neighbor in K to form a clique of size >= 2 with v
                clique = set(nbrs_k)
                clique.add(v)
                candidates.append(frozenset(clique))
            # If len(nbrs_k) == 0, v is isolated from K, no cross edges. No clique needed for v alone (size 1 not allowed).
            
        # Also, what if a clique consists of 2 vertices from K?
        # That is covered by the K clique.
        # What if a clique is {u, v} where u in K, v in I?
        # That is covered by C_v.
        
        # So the set of maximal cliques is sufficient.
        # Now we need to select a subset of these cliques that covers all edges in edge_set, minimizing the count.
        # This is a Set Cover problem.
        # |edge_set| is at most ~n^2/2. For n=90, ~4000 edges.
        # Number of candidates is at most n (1 for K + n-k for I).
        # Greedy Set Cover:
        # 1. Uncovered = set(edge_set)
        # 2. While Uncovered is not empty:
        #    a. Find candidate C that covers the most edges in Uncovered.
        #    b. Add C to solution.
        #    c. Remove edges covered by C from Uncovered.
        
        # To speed this up, we can precompute for each candidate which edges it covers.
        # But we can also just check on the fly.
        
        # Optimization: 
        # The K clique covers all edges within K.
        # C_v covers edges within N_K(v) and cross edges to v.
        
        # Let's implement the greedy selection.
        
        # Precompute edge lists for each candidate to avoid O(|C|^2) checks repeatedly?
        # Or just maintain a set of uncovered edges.
        
        uncovered_edges = set(edge_set)
        selected_cliques = []
        
        # Sort candidates by size descending? Might help tie-breaking or early pruning.
        # Actually, we compute "coverage" dynamically.
        
        # To make it faster, we can represent edges as a set of tuples.
        
        # Heuristic: 
        # If we pick K, we remove all internal K edges.
        # Then for each v in I, we must cover cross edges (u, v).
        # If we pick C_v, we cover cross edges (u, v) for all u in N_K(v), AND internal edges among N_K(v).
        # Note: If we already picked K, the internal edges among N_K(v) are already covered.
        # So if K is picked, C_v only helps cover cross edges to v.
        # If K is NOT picked, C_v helps cover internal edges too.
        
        # The greedy algorithm will handle this trade-off naturally if we calculate "newly covered edges" correctly.
        
        # Let's refine the candidate list. 
        # Do we need to consider ALL subsets? No, maximal is enough.
        
        # However, there is a subtlety. 
        # Is it possible that a non-maximal clique is better?
        # Suppose C1 and C2 overlap.
        # Using a subset of C1 might leave more edges for C2 to cover exclusively?
        # No, set cover is monotone. If A covers E_A and B covers E_B.
        # If we use A U {x}, it covers E_A U E_x.
        # It covers everything A covers. It cannot "uncover" anything.
        # So using a maximal clique is always at least as good as using a subset, in terms of edges covered.
        # The only reason to use a subset is if the maximal clique is "too big" and causes some penalty? 
        # No, the cost is 1 per clique regardless of size.
        # So yes, maximal cliques are strictly dominant.
        
        # One exception: If a maximal clique C covers an edge e, but e is already covered by another chosen clique, 
        # that's fine. The "benefit" of C is the number of *uncovered* edges it covers.
        
        # Implementation details:
        # 1. Create list of candidate cliques (as lists of vertices).
        # 2. Create a set of all edges.
        # 3. Loop:
        #    - For each candidate, calculate number of edges it covers that are currently in `uncovered_edges`.
        #    - Pick the one with max count.
        #    - If max count is 0, break (should not happen if graph is connected or we have all edges).
        #    - Add candidate to solution.
        #    - Remove its covered edges from `uncovered_edges`.
        
        # Complexity:
        # Num candidates M ~ n.
        # Num edges E ~ n^2.
        # Each step: M * E checks? 
        # n=90, M~60, E~4000. 60*4000 = 240,000 ops per step.
        # Steps ~ E / avg_size. Avg size of clique ~ k/2? ~15. Steps ~ 4000/15 ~ 260.
        # 260 * 240,000 = 62 million ops.
        # Python might be slow for 62M ops in a tight loop with set intersections.
        # Let's optimize the "count uncovered edges" step.
        
        # Instead of iterating all edges of candidate and checking membership in uncovered:
        # We can iterate `uncovered_edges` and check if they are in candidate?
        # Or precompute the set of edges for each candidate.
        
        candidate_edge_sets = []
        for c in candidates:
            v_list = list(c)
            c_edges = set()
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    if u > v: u, v = v, u
                    c_edges.add((u, v))
            candidate_edge_sets.append(c_edges)
            
        # Now greedy:
        # uncovered_edges is a set.
        # For each candidate i, count = len(uncovered_edges & candidate_edge_sets[i])
        
        # This intersection is efficient in Python sets.
        
        while uncovered_edges:
            best_idx = -1
            best_count = 0
            best_edge_set = None
            
            # Optimization: if time is running out, fall back to simple pairs?
            # But 60 candidates * 260 steps is small.
            
            for i in range(len(candidates)):
                # Check if this candidate still covers anything
                # Fast check: if candidate_edge_sets[i] is subset of already covered?
                # We don't track "covered" explicitly, only "uncovered".
                # So just intersect.
                count = len(uncovered_edges & candidate_edge_sets[i])
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_edge_set = candidate_edge_sets[i]
            
            if best_count == 0:
                # Should not happen if candidates cover all edges.
                # Fallback: add pairs for remaining edges
                for e in uncovered_edges:
                    selected_cliques.append(list(e))
                break
            
            selected_cliques.append(list(candidates[best_idx]))
            uncovered_edges.difference_update(best_edge_set)
            
            # Check time
            if time.time() > deadline - 10:
                # Fallback: cover remaining edges with pairs
                for e in uncovered_edges:
                    selected_cliques.append(list(e))
                break
        
        return selected_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        # Ensure types are correct
        # cliques is list of lists of ints.
        # ints are python ints from range().
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
