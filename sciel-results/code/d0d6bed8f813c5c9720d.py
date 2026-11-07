"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges in K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify K and I
        K = set(range(k))
        I = set(range(k, n))
        
        # Neighbors in K for each v in I
        neighbors = {v: set() for v in I}
        for v in I:
            for u in K:
                if (u, v) in edges or (v, u) in edges:
                    neighbors[v].add(u)
        
        # We need to cover all edges exactly once.
        # Edges are of two types:
        # 1. Internal K edges: (u, w) where u, w in K.
        # 2. Cross edges: (u, v) where u in K, v in I.
        
        # Strategy:
        # Each clique can contain at most one vertex from I.
        # So cliques are either:
        # a) Pure K cliques: subset of K
        # b) Mixed cliques: {v} U S, where v in I, S subset of K, and S is a clique in K (which it is, since K is a clique), 
        #    AND v is connected to all vertices in S.
        
        # If we use a mixed clique {v} U S, it covers:
        # - All cross edges (u, v) for u in S.
        # - All internal K edges (u, w) for u, w in S.
        
        # If we use a pure K clique S, it covers:
        # - All internal K edges (u, w) for u, w in S.
        
        # We want to minimize the total number of cliques.
        
        # Let's think about the internal K edges. There are k*(k-1)/2 of them.
        # If we use a mixed clique {v} U S, it "absorbs" |S|*(|S|-1)/2 internal edges.
        # If we use a pure K clique S, it "absorbs" |S|*(|S|-1)/2 internal edges.
        
        # The cross edges must be covered by mixed cliques.
        # For each v in I, let N(v) be its neighbors in K.
        # The cross edges incident to v are exactly (u, v) for u in N(v).
        # These must be covered by mixed cliques containing v.
        # Let the mixed cliques containing v be {v} U S_1, {v} U S_2, ..., {v} U S_m.
        # Then S_1 U S_2 U ... U S_m = N(v), and S_i are disjoint.
        # Also, each S_i must be a clique in K (always true) and v must be connected to all in S_i (true by definition of N(v)).
        # The internal edges covered by these mixed cliques are the edges induced by each S_i.
        
        # The remaining internal K edges must be covered by pure K cliques.
        
        # To minimize the number of cliques, we should try to make the S_i as large as possible, 
        # because a larger S_i covers more internal edges, reducing the need for pure K cliques.
        # However, the S_i for a fixed v must be disjoint.
        
        # Actually, since K is a complete graph, any subset of K is a clique.
        # So for a fixed v, we can partition N(v) into any number of subsets.
        # To minimize the total number of cliques, we want to maximize the "efficiency" of covering internal edges.
        # The number of internal edges covered by a subset S is |S|*(|S|-1)/2.
        # This is a convex function. To maximize the sum of convex functions given a fixed sum of sizes (which is |N(v)|), 
        # we should make the sizes as unequal as possible? No, we want to minimize the number of cliques.
        
        # Let's consider the total number of cliques.
        # Let m_v be the number of mixed cliques for vertex v.
        # Let P be the number of pure K cliques.
        # Total cliques = sum(m_v) + P.
        
        # The internal edges covered by mixed cliques are the union of edges induced by S_i for all v and all i.
        # Since the S_i for different v's are not necessarily disjoint, an internal edge (u, w) might be covered by a mixed clique for v1 and also be "available" for a mixed clique for v2?
        # No, the problem states "every edge must lie in exactly one listed clique".
        # So an internal edge (u, w) can be covered by AT MOST ONE mixed clique (or one pure K clique).
        
        # This means that for each internal edge (u, w), it can be assigned to at most one mixed clique {v} U S such that u, w in S.
        
        # This looks like a complex optimization problem. Let's try a greedy approach.
        
        # 1. For each v in I, we need to cover the cross edges (u, v) for u in N(v).
        #    We do this by partitioning N(v) into subsets S_1, ..., S_m.
        #    Each subset S_j defines a mixed clique {v} U S_j.
        #    The internal edges induced by S_j are "consumed" by this clique.
        
        # 2. After assigning all mixed cliques, some internal K edges remain uncovered.
        #    We cover these with pure K cliques.
        
        # How to choose the partition of N(v)?
        # If we make one large subset S = N(v), we get 1 mixed clique, which consumes |N(v)|*(|N(v)|-1)/2 internal edges.
        # If we split N(v) into two subsets S1, S2, we get 2 mixed cliques, which consume |S1|*(|S1|-1)/2 + |S2|*(|S2|-1)/2 internal edges.
        # Note that |N(v)|*(|N(v)|-1)/2 > |S1|*(|S1|-1)/2 + |S2|*(|S2|-1)/2 if S1, S2 are proper non-empty subsets.
        # So, making the subsets as large as possible (i.e., not splitting) covers MORE internal edges.
        # This is good, because it reduces the number of pure K cliques needed.
        # However, we must ensure that the internal edges consumed by different mixed cliques do not overlap.
        
        # So the conflict arises if we have two vertices v1, v2 in I, and we form mixed cliques {v1} U S1 and {v2} U S2, 
        # and S1 and S2 share some vertices. If S1 and S2 share vertices u, w, then the edge (u, w) is in both cliques, which is not allowed.
        
        # Therefore, the sets S (for all mixed cliques) must be pairwise disjoint on the internal edges.
        # Actually, if S1 and S2 share one vertex u, no internal edge is shared.
        # If S1 and S2 share two vertices u, w, then the edge (u, w) is shared.
        # So the condition is: for any two mixed cliques, their K-parts can share at most 1 vertex.
        
        # This is a strong constraint.
        
        # Let's try a different perspective.
        # We want to cover all edges.
        # Let's iterate through the vertices in I. For each v, we decide which K-vertices to include in the mixed clique(s) for v.
        
        # A simple greedy strategy:
        # For each v in I (in some order), look at the remaining uncovered cross edges (u, v) for u in N(v).
        # We want to form mixed cliques to cover these.
        # We can form one mixed clique {v} U S, where S is a subset of the available K-vertices that are connected to v.
        # To be safe and avoid conflicts, let's try to form as few mixed cliques as possible for each v, 
        # but ensure that the K-parts of mixed cliques for different v's don't overlap in edges.
        
        # Actually, if we just take S = N(v) for each v, and if the N(v)'s are disjoint, we are fine.
        # But they are not disjoint.
        
        # Let's try this:
        # For each v in I, we start with S = N(v).
        # We check if S conflicts with any previously formed mixed clique.
        # A conflict exists if there is a previously formed mixed clique {v'} U S' such that |S ∩ S'| >= 2.
        # If a conflict exists, we need to shrink S or split it.
        
        # This is getting complicated. Let's try a simpler, potentially suboptimal but valid approach first, 
        # and then see if we can improve.
        
        # Approach:
        # 1. For each v in I, create one mixed clique {v} U N(v).
        #    This covers all cross edges for v.
        #    It also covers all internal edges within N(v).
        # 2. Check for conflicts. If two mixed cliques {v1} U N(v1) and {v2} U N(v2) have |N(v1) ∩ N(v2)| >= 2, 
        #    then there is an edge (u, w) in the intersection that is covered by both. This is invalid.
        
        # So we need to resolve conflicts.
        # We can do this by iteratively removing edges from the "covered by mixed cliques" set.
        # When a conflict is detected, we "un-cover" one of the edges and assign it to a pure K clique later.
        # But we can't "un-cover" an edge from a clique without changing the clique.
        
        # Alternative:
        # We build the mixed cliques one by one.
        # Maintain a set of "forbidden" pairs in K. A pair (u, w) is forbidden if it is already covered by a mixed clique.
        # When forming a mixed clique for v, we choose a subset S of N(v) such that no pair in S is forbidden.
        # This means S can contain at most one vertex from any forbidden pair.
        
        # This is equivalent to finding a maximum independent set in a graph where the vertices are N(v) 
        # and edges connect forbidden pairs? No, it's finding a subset of N(v) that contains no forbidden pair.
        # This is the same as finding a clique in the complement of the forbidden graph restricted to N(v).
        
        # Let's implement this greedy approach.
        
        forbidden_pairs = set()
        mixed_cliques = []
        
        # Process vertices in I. Order might matter. Let's try sorting by degree in K (size of N(v)) descending.
        # Or maybe ascending? Let's try descending first.
        I_sorted = sorted(I, key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in I_sorted:
            if time.time() > deadline:
                break # Fallback needed, but let's hope it's fast enough.
                
            N_v = neighbors[v]
            if not N_v:
                continue
                
            # We want to partition N_v into subsets S_1, ..., S_m such that:
            # 1. Each S_j contains no forbidden pair.
            # 2. We want to minimize m (number of mixed cliques for v).
            # 3. We want to maximize the size of S_j to cover more internal edges (marking them as forbidden).
            
            # Greedy:
            remaining = set(N_v)
            new_sets = []
            
            while remaining:
                # Find a maximal subset S of remaining with no forbidden pairs.
                # This is the same as finding a maximal clique in the graph G_S where vertices are remaining
                # and edges exist between u, w if (u, w) is NOT forbidden.
                
                # Since K is small (k <= 30), we can do a simple greedy.
                # Pick a vertex, add others if compatible.
                
                # To be more strategic, pick the vertex with the most non-forbidden neighbors in remaining?
                # Or just pick the first one.
                
                # Let's try picking the vertex with the highest degree in the "non-forbidden" graph induced by remaining.
                
                # Actually, just picking the first one is O(k^2) which is fine.
                
                start = next(iter(remaining))
                S = {start}
                remaining.remove(start)
                
                for u in list(remaining):
                    can_add = True
                    for w in S:
                        if (min(u, w), max(u, w)) in forbidden_pairs:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                        remaining.remove(u)
                
                new_sets.append(S)
                
                # Mark all pairs in S as forbidden
                for u, w in combinations(S, 2):
                    forbidden_pairs.add((min(u, w), max(u, w)))
            
            # Add these mixed cliques
            for S in new_sets:
                if S:
                    mixed_cliques.append((v, S))
        
        # Now, construct the list of cliques
        cliques = []
        
        # Add Mixed Cliques
        for v, S in mixed_cliques:
            clique_verts = list(S) + [v]
            if len(clique_verts) >= 2:
                cliques.append(clique_verts)
        
        # Identify remaining internal edges in K
        remaining_edges = set()
        for u in K:
            for w in K:
                if u < w:
                    if (u, w) not in forbidden_pairs:
                        remaining_edges.add((u, w))
        
        # Cover remaining edges with Pure K cliques
        # Greedy Clique Cover
        rem_edges = set(remaining_edges)
        while rem_edges:
            if time.time() > deadline:
                break
            # Find a maximal clique in the graph of remaining edges
            u, v = next(iter(rem_edges))
            S = {u, v}
            rem_edges.discard((u, v))
            
            changed = True
            while changed:
                changed = False
                for w in K:
                    if w not in S:
                        ok = True
                        for x in S:
                            if (min(w, x), max(w, x)) not in rem_edges:
                                ok = False
                                break
                        if ok:
                            S.add(w)
                            for x in list(S):
                                if x != w:
                                    rem_edges.discard((min(w, x), max(w, x)))
                            changed = True
            
            if len(S) >= 2:
                cliques.append(list(S))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
