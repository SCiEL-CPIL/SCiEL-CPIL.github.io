"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    _deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve(n, k, p, seed):
        # Build graph
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        rng = random.Random(seed)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Build cross-adjacency: for each i in I, set of C-vertices adjacent
        cross = [set() for _ in I]
        for idx, i in enumerate(I):
            for u in C:
                if u in adj[i]:
                    cross[idx].add(u)

        # Greedy clique-partition: iteratively pick a clique subset of C and
        # assign all I-vertices whose cross-neighborhood is contained in that subset.
        remaining_C = set(C)
        remaining_I = list(range(len(I)))  # indices into I list
        cliques = []

        # Sort I-vertices by degree (ascending) to handle isolated ones first
        # Actually we'll handle them in the loop.

        # We need to keep iterating until no I-vertices remain unassigned
        # and remaining_C is empty (or only singletons left that can't form cliques)
        
        # Strategy: at each step, find a subset S of remaining_C that is a clique
        # (it always is, since C is a clique), and assign all I-vertices whose
        # cross-neighborhood ⊆ S. We want S to "cover" many I-vertices.
        # 
        # A good heuristic: for each candidate I-vertex, its cross-neighborhood
        # defines a minimal required subset of C. Group I-vertices with similar
        # neighborhoods.
        
        # Simpler approach: iteratively pick the I-vertex with the largest
        # cross-neighborhood, set S = its cross-neighborhood (plus maybe more),
        # and assign all I-vertices whose cross ⊆ S.
        
        # Even simpler and effective: 
        # 1. Find I-vertices with empty cross-neighborhood -> they can go anywhere.
        # 2. For the rest, use a greedy set-cover-like approach.
        
        # Let me use a different, cleaner approach:
        # Partition C into groups. For each I-vertex, it must be assigned to a group
        # that contains all its cross-neighbors. To minimize groups, we want to
        # group C-vertices that are "co-used" by many I-vertices.
        
        # Greedy: 
        # - Start with all C in one pool.
        # - Repeatedly: find the I-vertex with the largest |cross|, take S = cross[i],
        #   assign all I-vertices with cross ⊆ S to this group, remove S from pool,
        #   remove those I-vertices.
        # - Leftover I-vertices with cross ⊆ remaining pool: assign similarly.
        # - Any I-vertex with empty cross can be assigned to any group (or a new one with 2 C-verts).
        # - Remaining C-vertices: pair them up into 2-cliques.
        
        # Actually, let me think more carefully. The constraint is:
        # Each clique = S ∪ A where S ⊆ C (|S| ≥ 1, but clique needs ≥2 total vertices)
        # and A ⊆ I such that for all i in A, cross[i] ⊆ S.
        # Every edge must be in exactly one clique:
        # - C-C edges: each pair (u,v) in C is in exactly one clique, meaning u and v
        #   are in the same S for exactly one group. So C is partitioned into groups.
        # - C-I edges: edge (u, i) is in the clique assigned to i, and u must be in that clique's S.
        #   Since cross[i] ⊆ S_assigned(i), all edges from i to C are covered.
        #   Edge (u,i) appears in exactly one clique (the one for i). Good.
        # - I-I edges: none exist (I is independent). Good.
        #
        # So the problem reduces to: partition C into subsets S_1, ..., S_m, and assign
        # each i in I to some j such that cross[i] ⊆ S_j, minimizing m.
        # Constraint: each S_j must have |S_j| ≥ 1 (and if |S_j|=1, then |A_j| ≥ 1 to make clique size ≥ 2).
        # Actually if |S_j| = 1 and A_j is empty, that's a 1-vertex "clique" which is invalid.
        # So we need |S_j| + |A_j| ≥ 2 for each group.
        
        # This is a set-partition / covering problem. Let me use greedy.
        
        # Greedy algorithm:
        # 1. Sort I-vertices by |cross| descending.
        # 2. For each unassigned i, find the "best" S to assign it to:
        #    - If there's an existing group j with cross[i] ⊆ S_j, assign to j (prefer smallest S_j that works? or largest A_j?).
        #    - Otherwise, create a new group with S = cross[i] (if cross[i] is non-empty) or S = {some C-vertex} (if empty).
        # 3. But we also need to ensure C is fully partitioned (every C-vertex in exactly one S_j).
        
        # Hmm, the constraint that C is partitioned makes this trickier. Let me think again.
        
        # Alternative formulation: We need to partition C into groups S_1,...,S_m.
        # Each i must be assigned to a group j with cross[i] ⊆ S_j.
        # If no such group exists, the partition is invalid for that i.
        # So we need: for every i, ∃j: cross[i] ⊆ S_j.
        # This means: the union of all S_j = C (partition), and for each i, cross[i] is contained in some S_j.
        # Equivalently: for each i, cross[i] must not be "split" across multiple groups.
        
        # Greedy approach:
        # - Start: all C in one group. Check if all i can be assigned (yes, since cross[i] ⊆ C).
        # - Try to split: for a group S, can we split it into S1, S2 such that no i has cross[i] intersecting both S1 and S2 but not being contained in either?
        #   i.e., no i with cross[i] ∩ S1 ≠ ∅ and cross[i] ∩ S2 ≠ ∅ and cross[i] ⊄ S1 and cross[i] ⊄ S2.
        #   Actually we need: for each i, cross[i] ⊆ S1 or cross[i] ⊆ S2 (if cross[i] ⊆ S).
        #   This is a 2-coloring-like constraint.
        
        # This is getting complex. Let me use a simpler but effective greedy:
        
        # Approach: 
        # 1. Identify "conflict" structure: two I-vertices i, i' conflict if there's no single
        #    subset of C that can contain both cross[i] and cross[i'] while being a valid group.
        #    Actually, i and i' can share a group iff cross[i] ∪ cross[i'] is the S (or a superset).
        #    They can't share a group if... actually they can always share a group with S = cross[i] ∪ cross[i'].
        #    The issue is that C must be partitioned, so if i needs {1,2} and i' needs {2,3},
        #    they can share a group S={1,2,3}, or be in separate groups S1={1,2}, S2={2,3} — but wait,
        #    2 can't be in both groups! So they CAN'T be in separate groups if their crosses overlap.
        #    They MUST be in the same group if cross[i] ∩ cross[i'] ≠ ∅.
        #    
        #    Wait no: if cross[i] = {1,2} and cross[i'] = {2,3}, they could be in the same group S={1,2,3},
        #    or... i in group with S={1,2} and i' in group with S={2,3} — but 2 is in both, violating partition.
        #    So yes, if cross[i] ∩ cross[i'] ≠ ∅, they MUST be in the same group.
        
        # This means: the groups form a partition of C, and I-vertices are assigned to groups.
        # Two I-vertices with overlapping cross-neighborhoods must be in the same group.
        # This defines an equivalence relation: i ~ i' if cross[i] ∩ cross[i'] ≠ ∅ (transitive closure).
        # Wait, is it transitive? If cross[i]∩cross[i']≠∅ and cross[i']∩cross[i'']≠∅, must i and i'' be in same group?
        # cross[i] might be {1,2}, cross[i']={2,3}, cross[i'']={3,4}. i and i'' have disjoint crosses.
        # But i' must be in a group containing {2,3}. If i is in a group with S⊇{1,2} and i'' in group with S'⊇{3,4},
        # then i' can't be in either (needs 2 and 3). So i' must be in a third group, or i and i'' must be in the same group as i'.
        # Actually: i' needs a group with S ⊇ {2,3}. i needs S ⊇ {1,2}. i'' needs S ⊇ {3,4}.
        # Option: all three in one group S={1,2,3,4}. Or: i in group1 S1⊇{1,2}, i' in group2 S2⊇{2,3} — impossible since 2 in both.
        # So yes, transitive: if there's a chain of overlapping crosses, all must be in the same group.
        
        # Therefore: the connected components of the graph on I (edges between i,i' if cross[i]∩cross[i']≠∅)
        # must each be assigned to a single group. And the group's S must contain the union of crosses of all i in that component.
        
        # So the minimum number of groups ≥ number of connected components.
        # And we can achieve exactly that by: for each component, S = union of crosses, assign all i in component to that group.
        # But we also need to handle: C-vertices not in any cross (isolated in cross graph) — they can be paired arbitrarily.
        # And I-vertices with empty cross — they can go in any group.
        
        # Wait, but this might not be minimal! Because within a component, the union of crosses might be large,
        # but maybe we can split the component into multiple groups if the overlap structure allows.
        # Example: cross[i1]={1,2}, cross[i2]={2,3}, cross[i3]={3,4}. All connected.
        # Union = {1,2,3,4}. One group. But could we do: group1 S={1,2,3} for i1,i2, group2 S={3,4} for i3?
        # No! 3 is in both groups. Violates partition.
        # What about: group1 S={1,2} for i1, group2 S={2,3} for i2, group3 S={3,4} for i3? 2 and 3 each in two groups. No.
        # So indeed, the connected component must be in one group.
        
        # Hmm wait, that's not quite right either. Let me reconsider.
        # The constraint is: C is partitioned into S_1,...,S_m. Each i is assigned to exactly one j with cross[i] ⊆ S_j.
        # For i and i' to be in different groups j ≠ j', we need cross[i] ⊆ S_j and cross[i'] ⊆ S_j' with S_j ∩ S_j' = ∅.
        # This requires cross[i] ∩ cross[i'] = ∅.
        # So: if cross[i] ∩ cross[i'] ≠ ∅, they MUST be in the same group.
        # This is a necessary condition. Is it sufficient for the component to be in one group? Yes, because
        # if we put all i in a connected component into one group with S = ∪_{i in comp} cross[i], it works.
        # And we can't do better (fewer groups) because of the necessity.
        # But can we do better by NOT putting all in one group? No, because of the necessity.
        # So the minimum number of groups for I-vertices with non-empty cross = number of connected components.
        
        # But wait — can we merge components? If component A has union U_A and component B has union U_B with U_A ∩ U_B = ∅,
        # we could put them in the same group S = U_A ∪ U_B. That would reduce group count!
        # So the real problem is: partition C into groups, where each group's S must contain the union of crosses
        # of all I-vertices assigned to it, and each connected component must be entirely in one group.
        # To minimize groups, we want to merge components whose union-of-crosses are disjoint.
        
        # So: 
        # 1. Find connected components of I (non-empty cross) via cross-overlap.
        # 2. For each component c, compute U_c = union of crosses.
        # 3. Now we need to partition {U_c} into groups where each group's sets are pairwise disjoint,
        #    plus handle remaining C-vertices not in any U_c.
        #    This is a set-packing / graph coloring problem on the components (conflict if U_c ∩ U_d ≠ ∅).
        #    But we already established that if U_c ∩ U_d ≠ ∅, they must be in the same group (since there's a path of overlaps).
        #    Wait, no: U_c ∩ U_d ≠ ∅ means some C-vertex is in both unions, so they CAN'T be in separate groups.
        #    So components with overlapping U's must be merged. This is the same connected-component logic applied to U's.
        #    But we already took connected components based on individual cross overlaps, which implies U overlaps.
        #    Actually, two components could have overlapping U's without any individual cross overlap? 
        #    No: if U_c ∩ U_d ≠ ∅, there exist i in c, i' in d with cross[i] ∩ cross[i'] ≠ ∅ (since U_c = ∪ crosses in c).
        #    So they'd be in the same connected component. Contradiction. So different components have disjoint U's.
        #    
        #    Therefore: each connected component has a disjoint U_c, and they can each be their own group,
        #    or we can merge multiple components into one group (since their U's are disjoint, merging is fine).
        #    To minimize groups, we should merge as many as possible! But merging doesn't help if we're counting groups —
        #    merging two components into one group reduces group count by 1. So we want to merge ALL components into one group?
        #    But then S = ∪ all U_c, and we assign all I-vertices to that one group. That's just 1 group for all non-empty-cross I-vertices!
        #    Plus we need to handle C-vertices not in any U_c.
        
        # Wait, that can't be right. If all I-vertices go in one group with S = C (all of C), then we have 1 group.
        # Plus leftover C-vertices... but if S = C, there are no leftover C-vertices.
        # So the answer would be: 1 group containing all of C and all I-vertices with non-empty cross,
        # plus groups for I-vertices with empty cross (they can join the main group!),
        # plus... wait, if S = C, then all I-vertices (empty or not) can be assigned to this group.
        # So the entire graph is covered by ONE clique: C ∪ I? But I is an independent set, so C ∪ I is not a clique!
        
        # I made an error. The clique is S ∪ A where S ⊆ C and A ⊆ I. For this to be a clique,
        # every pair in S∪A must be adjacent. Pairs in S: adjacent (C is clique). Pairs in A: NOT adjacent (I is independent)!
        # So |A| ≤ 1! Each group can contain at most ONE I-vertex!
        
        # Oh no, I completely missed this. Since I is an independent set, no two I-vertices can be in the same clique.
        # So each clique contains at most one I-vertex.
        # 
        # Revised problem:
        # - C-C edges: partition C into groups S_1,...,S_m. Each pair in S_j is covered by clique j.
        # - C-I edges: each i in I is assigned to a group j with cross[i] ⊆ S_j. The clique is S_j ∪ {i}.
        #   This covers all edges from i to C (since cross[i] ⊆ S_j).
        # - Each clique S_j ∪ A_j where A_j ⊆ I, |A_j| ≤ 1.
        # - Constraint: |S_j| + |A_j| ≥ 2.
        # - Every C-C edge covered exactly once: C is partitioned.
        # - Every C-I edge covered exactly once: each i assigned to exactly one group.
        
        # So: we partition C into m groups. Each i is assigned to one group j with cross[i] ⊆ S_j.
        # Minimize m.
        # Constraint: if S_j is a singleton {u} and no i is assigned to j, then clique size = 1 < 2. Invalid.
        # So singleton S_j must have at least one i assigned.
        
        # Now the problem: minimize m (number of groups = partition of C).
        # Each i must be "hosted" by a group whose S contains cross[i].
        
        # Lower bound: m ≥ max over all i of |cross[i]| (since S_j must contain cross[i], |S_j| ≥ |cross[i]|,
        # and sum of |S_j| = k, so m ≤ k but also each S_j can host at most... hmm).
        # Actually m ≥ ceil(k / max_j |S_j|) but that's not directly helpful.
        # m ≥ max_i |cross[i]| is NOT a lower bound on m; it's a lower bound on the size of the group hosting i.
        
        # The real constraint: we need to partition C into m subsets such that for each i,
        # there exists j with cross[i] ⊆ S_j.
        
        # Greedy: 
        # 1. Sort I by |cross| descending.
        # 2. For each i, it needs a group with S ⊇ cross[i]. 
        # 3. We want to minimize the number of groups.
        # 4. A group can host multiple i's as long as their crosses are all ⊆ S_j.
        
        # This is similar to: given sets cross[i] ⊆ C, partition C into minimum groups such that
        # each cross[i] is contained in some group.
        
        # Greedy heuristic:
        # - Start with all C unassigned to groups.
        # - Process i in order of |cross[i]| descending.
        # - For current i, try to find an existing group j with cross[i] ⊆ S_j. Assign i to j.
        # - If no such group, create a new group with S = cross[i] (and any unassigned C-vertices to fill it? no, S must be exactly what we assign).
        #   Actually, we're partitioning C, so S_j are the parts. We can't just pick S = cross[i] if some vertices in cross[i] are already in another group.
        #   
        # Let me think differently. Since we're partitioning C, each C-vertex goes to exactly one group.
        # The constraint is: for each i, all vertices in cross[i] must be in the SAME group.
        # This is a constraint satisfaction / graph partitioning problem.
        # Define a graph G on C: edge (u,v) if ∃i with u,v ∈ cross[i]. Then u and v must be in the same group.
        # The connected components of G must each be in one group.
        # But we can merge multiple components into one group.
        # To minimize groups, merge all components into one group? Then S_1 = C, and all i are assigned to group 1.
        # But then we have 1 group, and all I-vertices assigned to it. Clique = C ∪ {i} for each i. That's |I| cliques, not 1!
        
        # I'm confusing myself. Let me re-read the problem.
        # "Every listed set must be a clique" — each listed set is one clique.
        # "every edge must lie in exactly one listed clique" — edge coverage.
        # "Minimize the total number of cliques" — minimize count of listed sets.
        
        # So if I have C = {1,2,3,4,5} and I = {6,7,8} with various cross edges,
        # I might list cliques like: [1,2,3,6], [4,5,7], [8, ...] — wait, [8] alone is size 1, invalid.
        # [8, 1]? But 1 is already in [1,2,3,6]. Edge (1,8) would be in both? No — each edge in exactly one clique.
        # If 1 is in clique [1,2,3,6], then edge (1,2) is in that clique. Edge (1,8) must be in some clique containing both 1 and 8.
        # But 1 is already "used up" in the sense that all its C-C edges are covered by [1,2,3,6].
        # Edge (1,8) is a C-I edge. It must be in a clique containing 1 and 8. The only clique containing 1 is [1,2,3,6] (since C is partitioned).
        # So 8 must be in [1,2,3,6], meaning 8's cross-neighborhood must be ⊆ {1,2,3}.
        
        # So: each C-vertex belongs to exactly one clique (its group). Each I-vertex belongs to exactly one clique (the one hosting it).
        # The clique hosting i must contain all of i's cross-neighbors.
        # 
        # Number of cliques = number of groups (each group S_j hosts some subset A_j of I, |A_j| ≤ 1).
        # Wait, |A_j| ≤ 1 because I is independent. So each group hosts at most one I-vertex.
        # Number of cliques = number of non-empty (S_j, A_j) pairs where |S_j| + |A_j| ≥ 2.
        # Since |A_j| ≤ 1, we need |S_j| ≥ 1 (and if |S_j| = 1, need |A_j| = 1; if |S_j| ≥ 2, A_j can be empty).
        
        # To minimize cliques: we want to minimize the number of groups. But each group can host at most 1 I-vertex.
        # If |I| > 0, we need at least |I| groups? No! A group with |S_j| ≥ 2 can have A_j = ∅ (no I-vertex),
        # and a group with |S_j| = 1 must have |A_j| = 1.
        # But we need to assign ALL I-vertices. Each I-vertex needs its own group (since |A_j| ≤ 1).
        # So we need at least |I| groups that each host one I-vertex.
        # Plus possibly additional groups for C-vertices not covered by those groups.
        
        # Wait, that means minimum cliques ≥ |I| = n - k? That seems large.
        # For n=90, k=30: |I| = 60. So at least 60 cliques.
        # The seed has: C-C edges = k(k-1)/2 = 435, plus cross edges ≈ p*k*|I| = 0.7*30*60 = 1260. Total ≈ 1695.
        # So 60 is much better than 1695.
        
        # But can we do better than |I|? No, because each I-vertex needs its own clique (can't share with another I-vertex).
        # So minimum = |I| + (extra groups needed for leftover C-vertices).
        
        # To minimize extra groups: each I-vertex i is assigned to a group with S ⊇ cross[i].
        # The union of all S_j (for groups hosting I-vertices) must cover all C-vertices that have at least one cross-edge.
        # C-vertices with no cross-edges can be in any group (or paired together).
        
        # Strategy:
        # 1. Assign each i to a group. The group's S must contain cross[i].
        # 2. To minimize total groups, we want the S_j's to cover all of C with as few groups as possible,
        #    but each group is tied to one I-vertex.
        # 3. Since each group hosts exactly one I-vertex (for the |I| groups), the S_j's form a partition of C
        #    (some groups may have empty A if |S_j| ≥ 2, but we need exactly |I| groups with A_j non-empty).
        #    Wait, we can have MORE than |I| groups: some groups with A_j = ∅ and |S_j| ≥ 2.
        #    But that increases the count. So we want exactly |I| groups if possible, with the S_j's partitioning C.
        
        # Can we always partition C into |I| subsets S_1,...,S_|I| such that cross[i] ⊆ S_i for each i?
        # This requires: for each i, |cross[i]| ≤ |S_i|, and the S_i's partition C.
        # Sum |S_i| = k. So we need sum |cross[i]| ≤ k? No, that's not the right constraint.
        # The constraint is: cross[i] ⊆ S_i, so S_i must contain cross[i]. The S_i's are disjoint and cover C.
        # This is possible iff: for any two i, i', cross[i] ∩ cross[i'] = ∅ (since S_i ∩ S_i' = ∅).
        # But that's not generally true! Cross-neighborhoods can overlap.
        
        # If cross[i] ∩ cross[i'] ≠ ∅, then i and i' can't be in separate groups (since S_i and S_i' are disjoint).
        # But we established each group hosts at most one I-vertex. Contradiction!
        # 
        # Wait, this means: if two I-vertices share a cross-neighbor, we CAN'T assign them to separate groups.
        # But we MUST assign them to separate groups (|A_j| ≤ 1). Contradiction!
        # 
        # Unless... the shared neighbor is in a group that hosts NEITHER of them. But the shared neighbor u is in cross[i] and cross[i'],
        # so u must be in the group hosting i AND in the group hosting i'. But u can only be in one group. Contradiction.
        
        # So the problem is INFEASIBLE if two I-vertices share a cross-neighbor?!
        # That can't be right. Let me re-examine.
        
        # Oh wait. I think I'm wrong about |A_j| ≤ 1. Let me re-check.
        # A clique S ∪ A where S ⊆ C, A ⊆ I. For this to be a clique, every pair must be an edge.
        # Pairs in A: (i, i') for i, i' ∈ A. Is (i, i') an edge? I is an independent set, so NO.
        # So indeed |A| ≤ 1. My analysis is correct.
        
        # So if two I-vertices share a cross-neighbor, the problem is infeasible?
        # That seems wrong. Let me check with a small example.
        # C = {1, 2}, I = {3, 4}. Edges: (1,3), (1,4), (2,3). 
        # Cross: cross[3] = {1,2}, cross[4] = {1}.
        # We need to cover edges: (1,2), (1,3), (1,4), (2,3).
        # Option: clique1 = [1,2,3] (covers (1,2),(1,3),(2,3)), clique2 = [1,4]? But 1 is in clique1 already.
        # Edge (1,4) must be in a clique with both 1 and 4. If 1 is in clique1, 4 must be in clique1. But clique1 = [1,2,3,4] is not a clique (3,4 not adjacent).
        # So infeasible? But the problem says split graphs can be partitioned...
        # 
        # Wait, I think the issue is that C-vertices can be in MULTIPLE cliques! Let me re-read.
        # "every edge must lie in exactly one listed clique" — edges are unique to one clique.
        # But vertices can be in multiple cliques? Let me check: "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        # It doesn't say vertices must be in exactly one clique! Only edges.
        # 
        # So a vertex can appear in multiple cliques, as long as each edge is in exactly one.
        # This changes everything!
        
        # Revised: We need a set of cliques (each size ≥ 2) such that every edge is in exactly one clique.
        # Vertices can be in multiple cliques.
        # 
        # For C-C edges: each pair (u,v) in C must be in exactly one clique. 
        # For C-I edges: each (u,i) must be in exactly one clique.
        # 
        # A clique can contain multiple C-vertices and at most one I-vertex (since I is independent).
        # So a clique is either:
        #   - A subset of C of size ≥ 2 (covers C-C edges within it)
        #   - A subset of C of size ≥ 1 plus one I-vertex i (covers C-C edges within the C-subset, and C-I edges from i to the C-subset)
        # 
        # To cover all C-C edges: the C-subsets across all cliques must form an edge-cover of the complete graph on C,
        # with each edge in exactly one clique. This means the C-subsets form a partition of the edges of K_k.
        # Each C-subset of size s covers C(s,2) edges.
        # 
        # To cover all C-I edges: for each i, the C-I edges from i must be partitioned among cliques containing i.
        # But i can be in at most... well, i can be in multiple cliques, but each clique containing i also contains a C-subset,
        # and the C-I edges from i are to the C-vertices in that subset. Since each C-I edge is in exactly one clique,
        # the C-subsets in cliques containing i must partition cross[i] (the set of C-neighbors of i).
        # 
        # So for each i: cross[i] is partitioned into subsets, each being the C-part of a clique containing i.
        # And for C-C edges: the C-subsets (from all cliques, whether or not they contain an I-vertex) must partition the edges of K_k.
        
        # This is a complex combinatorial design problem. Let me think about a good heuristic.
        
        # Simple approach: 
        # - For each i, create one clique: cross[i] ∪ {i} (if |cross[i]| ≥ 1). This covers all C-I edges from i.
        #   But the C-C edges within cross[i] are also covered by this clique.
        # - Remaining C-C edges (those not within any cross[i]) need to be covered by C-only cliques.
        # - C-only cliques: partition the remaining edges of K_k into cliques (edge-disjoint).
        #   A clique on a subset S covers all edges within S.
        #   To cover remaining edges: find a clique partition of the remaining graph.
        #   The remaining graph is K_k minus the edges covered by the I-cliques.
        #   An edge (u,v) is covered by an I-clique if ∃i with u,v ∈ cross[i].
        #   So remaining edges = {(u,v) : ∀i, u,v ∉ cross[i] simultaneously} = edges not in any cross[i].
        
        # Number of cliques = |{i : |cross[i]| ≥ 1}| + (number of C-only cliques needed).
        # 
        # To minimize: we want to maximize the C-C edges covered by I-cliques (reducing C-only cliques needed),
        # but we're constrained to one clique per i (cross[i] ∪ {i}).
        # Actually, we could split cross[i] into multiple parts, each forming a clique with i.
        # But that increases the number of cliques for i. Trade-off.
        
        # For now, let me use the simple approach: one clique per i (if non-empty cross), plus C-only cliques for remaining edges.
        
        # For C-only cliques: the remaining graph on C. We need to partition its edges into cliques.
        # Greedy: repeatedly find a maximal clique in the remaining graph, remove its edges, repeat.
        # Or: since the remaining graph is a subgraph of K_k, we can use a greedy edge-clique partition.
        
        # Let me implement this.
        
        # Build remaining edge set on C
        covered_CC = set()
        for i in I:
            ci = cross[I.index(i)]  # cross-neighborhood of i
            if len(ci) >= 1:
                for u, v in combinations(sorted(ci), 2):
                    covered_CC.add((u, v))
        
        remaining_edges = set()
        for u in C:
            for v in range(u + 1, k):
                if (u, v) not in covered_CC:
                    remaining_edges.add((u, v))
        
        # Greedy clique partition of remaining_edges
        # Build adjacency for remaining graph
        rem_adj = [set() for _ in range(k)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        c_only_cliques = []
        remaining = set(remaining_edges)
        while remaining:
            # Find a maximal clique greedily
            # Pick a vertex with highest degree in remaining
            if not remaining:
                break
            # Get vertices involved
            verts_in_rem = set()
            for u, v in remaining:
                verts_in_rem.add(u)
                verts_in_rem.add(v)
            # Pick vertex with max degree
            best_v = max(verts_in_rem, key=lambda x: len(rem_adj[x] & verts_in_rem))
            # Greedily build clique
            clique = {best_v}
            candidates = set(rem_adj[best_v] & verts_in_rem)
            while candidates:
                # Pick candidate with max common neighbors in candidates
                best_c = max(candidates, key=lambda x: len(rem_adj[x] & candidates))
                clique.add(best_c)
                candidates = candidates & rem_adj[best_c]
            # Remove edges within clique
            clique_list = sorted(clique)
            for u, v in combinations(clique_list, 2):
                if (u, v) in remaining:
                    remaining.discard((u, v))
            if len(clique_list) >= 2:
                c_only_cliques.append(clique_list)
            # Update rem_adj
            for u in clique_list:
                for v in clique_list:
                    if u != v and v in rem_adj[u]:
                        rem_adj[u].discard(v)
        
        # Build final clique list
        result = []
        for idx, i in enumerate(I):
            ci = sorted(cross[idx])
            if len(ci) >= 1:
                result.append(ci + [i])
        result.extend(c_only_cliques)
        
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > _deadline - 5:
            # Fallback: just use edges as 2-cliques
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
        else:
            results.append(_solve(n, k, p, seed))
    
    # Convert to proper types
    final = []
    for r in results:
        final.append([list(c) for c in r])
    return final
# EVOLVE-BLOCK-END
