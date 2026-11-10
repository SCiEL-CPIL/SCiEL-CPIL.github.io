"""Clique partition of split graphs using structural properties."""
import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency for V (clique part) and cross-neighborhoods
        V = set(range(k))
        I = set(range(k, n))
        
        # Adjacency within V (should be complete)
        V_adj = {v: set(V - {v}) for v in V}
        
        # Cross neighbors: for each i in I, N(i) subset of V
        cross_N = {i: set() for i in I}
        for u, v in edges:
            if u < k and v >= k:
                cross_N[v].add(u)
            elif u >= k and v < k:
                cross_N[u].add(v)
        
        cliques = []
        
        # Strategy: 
        # 1. For each i in I, we may form clique {i} ∪ S where S ⊆ N(i)
        # 2. This covers cross edges (i,s) for s in S, AND internal edges within S
        # 3. Remaining internal edges not covered by any cross-clique must be covered by V-cliques
        # 4. We want to minimize total number of cliques
        
        # Greedy: for each i, decide how much of N(i) to "absorb"
        # If we absorb S, we save |S| separate 2-cliques for cross edges, 
        # but we "use up" |S|*(|S|-1)/2 internal edges (covered for free)
        # The cost is: 1 clique for {i}∪S vs |S| cliques for individual cross edges
        # So it's always beneficial to absorb at least 2 vertices (saves at least 1 clique)
        # Absorbing 1 vertex: 1 clique for {i,v} vs 1 clique for cross edge — tie, but 
        #   it also covers 0 internal edges, so no benefit
        # Absorbing 0 vertices: no cross clique needed if no cross edges... but we need 
        #   to cover cross edges somehow
        
        # Key insight: every cross edge (i,v) MUST be in some clique. The only cliques 
        # that can contain (i,v) are {i,v} or {i,v}∪{other V vertices adjacent to i}.
        # Since V is a clique, {i}∪S is a clique iff S ⊆ N(i).
        
        # Optimal: for each i, pick ONE subset S_i ⊆ N(i) and form clique {i}∪S_i.
        # This covers all cross edges (i,v) for v in S_i. For v in N(i)\S_i, we need 
        # separate 2-cliques {i,v}.
        # Internal edges within S_i are covered. Internal edges not in any S_i must be 
        # covered by additional V-cliques.
        
        # To minimize: maximize total |S_i| to cover more cross edges with fewer cliques,
        # but the internal edges within S_i are "wasted" (they'd be covered by V-cliques anyway).
        # Actually no: if we DON'T include v in S_i, the edge (i,v) needs its own clique.
        # If we DO include v in S_i, the edge (i,v) is covered by the {i}∪S_i clique.
        # The internal edges within S_i are also covered by that clique, so they can't 
        # be in the V-clique. But if we had a V-clique covering all of V, those edges 
        # would be covered there too — double cover!
        
        # So the constraint: each internal edge must be in EXACTLY ONE clique.
        # If {i}∪S_i covers internal edges within S_i, then those edges CANNOT be in 
        # any V-clique.
        
        # Revised strategy:
        # - Choose subsets S_i for each i
        # - Cliques: {i}∪S_i for each i with S_i non-empty
        # - Plus 2-cliques {i,v} for each i, v in N(i)\S_i
        # - Plus cliques covering internal edges NOT in any S_i
        # - Minimize total count
        
        # Greedy approach: for each i, S_i = N(i) (absorb everything). This gives:
        #   - |I| cliques of form {i}∪N(i) (for i with N(i) non-empty)
        #   - 0 cross 2-cliques
        #   - Internal edges not in any N(i) must be covered
        #   - But internal edges within N(i) are covered by {i}∪N(i)
        #   - If two i's share a vertex v, then edges within {v}∪... are fine, but 
        #     internal edges between vertices in N(i1) ∩ N(i2) might be covered twice!
        
        # CRITICAL: if v ∈ N(i1) and v ∈ N(i2), and we form both {i1}∪N(i1) and 
        # {i2}∪N(i2), then any internal edge (v,w) where w ∈ N(i1)∩N(i2) is covered 
        # by BOTH cliques. That's invalid!
        
        # So we need: for any internal edge (u,v), it can be in at most one cross-clique.
        # An internal edge (u,v) is in clique {i}∪S_i iff u,v ∈ S_i.
        # So we need: for each internal edge (u,v), at most one i has {u,v} ⊆ S_i.
        
        # This is getting complex. Let me use a simpler but effective approach:
        # 
        # Approach: 
        # 1. Start with no cross-cliques. All cross edges need 2-cliques: E_cross cliques.
        # 2. All internal edges need coverage: use one V-clique: 1 clique.
        #    Total: E_cross + 1.
        # 3. For each i, consider "upgrading" to {i}∪S where S ⊆ N(i).
        #    This replaces |S| cross 2-cliques with 1 clique, saving |S|-1.
        #    But it "consumes" internal edges within S, which were covered by the V-clique.
        #    We must remove those from the V-clique and cover them separately.
        #    If the V-clique is split, we may need more cliques for remaining internal edges.
        #
        # Better: Don't use a single V-clique. Instead, build V-cliques and cross-cliques 
        # together, ensuring no double coverage.
        
        # Clean approach: 
        # - For each i in I (sorted by |N(i)| descending), form clique {i} ∪ N(i).
        #   Mark all internal edges within N(i) as "covered".
        # - After processing all i, cover remaining internal edges with greedy maximal cliques.
        # - This ensures each internal edge is covered at most once (by the first i that 
        #   includes both endpoints in N(i)), and cross edges are covered exactly once.
        
        covered_internal = set()
        
        # Sort I by |N(i)| descending to maximize absorption early
        i_list = sorted([i for i in I if cross_N[i]], key=lambda x: len(cross_N[x]), reverse=True)
        
        for i in i_list:
            N_i = cross_N[i]
            if not N_i:
                continue
            # Greedily build S ⊆ N_i such that no internal edge within S is already covered
            S = set()
            for v in sorted(N_i):
                # Check if adding v to S would cause double-coverage of any internal edge
                ok = True
                for w in S:
                    e = (min(v,w), max(v,w))
                    if e in covered_internal:
                        ok = False
                        break
                if ok:
                    S.add(v)
            
            if S:
                clique = list(S) + [i]
                cliques.append(clique)
                # Mark internal edges within S as covered
                S_list = list(S)
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        u, v = S_list[a_idx], S_list[b_idx]
                        covered_internal.add((min(u,v), max(u,v)))
            
            # For vertices in N_i not in S, we still need to cover cross edge (i,v)
            # These will be covered as 2-cliques {i, v}
            for v in N_i:
                if v not in S:
                    cliques.append([i, v])
        
        # Cover remaining internal edges
        # Build set of uncovered internal edges
        all_internal = set()
        for u in range(k):
            for v in range(u+1, k):
                all_internal.add((u, v))
        
        remaining_internal = all_internal - covered_internal
        
        # Greedy maximal clique extraction on remaining_internal
        rem_edges_set = set(remaining_internal)
        
        while rem_edges_set:
            if time.time() > deadline - 5:
                # Emergency: just add 2-cliques for remaining
                for u, v in rem_edges_set:
                    cliques.append([u, v])
                break
            
            # Pick first edge, expand to maximal clique
            u0, v0 = next(iter(rem_edges_set))
            
            candidate_clique = [u0, v0]
            clique_verts = {u0, v0}
            
            for w in range(k):
                if w in clique_verts:
                    continue
                # Check all edges from w to candidate_clique are in rem_edges_set
                ok = True
                for v in candidate_clique:
                    e = (min(w,v), max(w,v))
                    if e not in rem_edges_set:
                        ok = False
                        break
                if ok:
                    candidate_clique.append(w)
                    clique_verts.add(w)
            
            if len(candidate_clique) >= 2:
                cliques.append(candidate_clique)
                # Remove all edges within candidate_clique from rem_edges_set
                for a_idx in range(len(candidate_clique)):
                    for b_idx in range(a_idx+1, len(candidate_clique)):
                        u, v = candidate_clique[a_idx], candidate_clique[b_idx]
                        rem_edges_set.discard((min(u,v), max(u,v)))
            else:
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
