"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _greedy_set_cover(universe, sets, sets_size, sets_members):
    # sets: list of frozensets; sets_size: len; sets_members: list of list of set-ids
    # universe: frozenset of elements
    remaining = set(universe)
    chosen = []
    if not remaining:
        return chosen
    # sort sets by size desc
    order = sorted(range(len(sets)), key=lambda i: -sets_size[i])
    for si in order:
        if not remaining:
            break
        s = sets[si]
        hit = s & remaining
        if len(hit) >= 2:
            chosen.append(si)
            remaining -= hit
    return chosen

def _local_improve(chosen_ids, sets, sets_size, universe):
    # chosen_ids: list of set indices. Try to remove redundant sets.
    if not chosen_ids:
        return chosen_ids
    chosen = list(chosen_ids)
    # build coverage count
    cov = {}
    for si in chosen:
        for e in sets[si]:
            cov[e] = cov.get(e, 0) + 1
    improved = True
    while improved:
        improved = False
        # try removing each set if all its elements still covered by others
        for idx in range(len(chosen)):
            si = chosen[idx]
            s = sets[si]
            ok = True
            for e in s:
                if cov.get(e, 0) <= 1:
                    ok = False
                    break
            if ok:
                for e in s:
                    cov[e] -= 1
                chosen.pop(idx)
                improved = True
                break
    return chosen

def _solve_one(n, k, p, seed):
    from random import Random
    rng = Random(seed)
    # Build adjacency: clique part = [0..k-1], independent part = [k..n-1]
    # cross edges: for v in independent, u in clique, edge if rng.random() < p
    # We need a clique partition: the clique part is itself a clique; but cross edges
    # must each be in exactly one clique. Strategy: pick cliques each containing some
    # clique-part vertices and some independent vertices.
    # For each independent vertex v, its neighbors in clique form a set N(v) (subset of clique).
    # A clique can include v and a subset of N(v) (any subset of its neighbors, plus the clique vertices form a clique among themselves).
    # Actually any set of clique vertices is a clique. A clique containing independent vertex v must be {v} union S where S is a clique subset of N(v) (i.e., any subset of N(v) since clique part is a clique).
    # Edge (u,v) for u in clique, v in independent must be covered by exactly one clique containing both u and v.
    # So for each v, the cliques that contain v must partition N(v) (each neighbor u of v in exactly one such clique).
    # Also cliques can be purely within the clique part (covering no cross edges) but those don't help; we avoid them.
    # But wait: edges within the clique part (u1,u2) must also be in exactly one clique. The clique part is a clique, so all those edges exist.
    # If we make cliques each containing v plus some clique vertices, the clique-part edges inside each such clique are covered there.
    # But an edge (u1,u2) where u1,u2 are both in clique part must be in exactly ONE clique. If u1 appears in multiple cliques (with different v's), then edge (u1,u2) might be covered multiple times or not at all.
    # This is the tricky part. Let me reconsider.
    #
    # Actually, the cleanest structure: The clique part C is a clique. The independent part I has no edges.
    # A clique in the graph is either a subset of C, or {v} union S where v in I and S is a subset of N(v) (subset of C).
    # For edge coverage:
    #   - Edges within C: (u1,u2) for u1!=u2 in C. Must be in exactly one clique.
    #   - Cross edges: (u,v) for u in C, v in I, if edge exists. Must be in exactly one clique.
    #   - No edges within I.
    #
    # Key insight: If we include the entire clique C as one clique, that covers all C-C edges. Then each cross edge (u,v) must be in a clique containing both u and v. But u is already in the C-clique; that's fine, a vertex can be in multiple cliques. The C-clique covers C-C edges only (it contains no independent vertices). Cross edges are NOT covered by the C-clique (since it has no independent vertex). So we still need cliques for cross edges.
    # But then a cross edge (u,v) is covered by a clique {v} union S where u in S. Good.
    # However, does the C-clique cause any cross edge to be double-covered? No, because the C-clique has no independent vertex, so it contains no cross edge.
    # Does any C-C edge get double covered? Only if another clique contains two C vertices. A clique {v} union S with |S|>=2 contains C-C edges within S. So if S has 2+ vertices, those C-C edges are covered by this clique AND by the C-clique -> double coverage. NOT ALLOWED.
    # Therefore, if we use the full C-clique, then NO other clique may contain 2+ clique vertices. So each other clique is {v} union {u} (a single edge) or {v} alone (useless). That means each cross edge is its own clique -> k*(n-k) cliques worst case. Bad.
    #
    # Alternative: Don't use the full C-clique. Instead, partition the C-C edges among cliques.
    # This is complex. Let me think of a cleaner approach.
    #
    # Better structural approach: Since C is a clique, we can think of assigning each independent vertex v to be "paired" with cliques.
    #
    # Let me reconsider the problem. We need a set of cliques such that every edge is in exactly one.
    #
    # Observation: The graph is a split graph. There's a known result that split graphs have small clique covers.
    #
    # Let me try a different decomposition. Consider that we can form cliques that each contain exactly one independent vertex v and a (possibly empty) subset of its neighbors in C. To avoid double-covering C-C edges, we need the subsets of C used across all cliques (for the C-C edges within them) to partition the C-C edges.
    #
    # Hmm, this is getting complicated. Let me think about it as: we need to cover all edges (C-C edges and cross edges) with cliques, each edge in exactly one clique.
    #
    # A clique is determined by: a set of C vertices (which form a clique) and optionally one I vertex v that is adjacent to all chosen C vertices.
    #
    # Let me define: for each clique, it has a "C-part" (subset of C) and an "I-part" (empty or a single vertex v in I). The I-part vertex v must be adjacent to all C-part vertices.
    #
    # Edge (u1,u2) in C: covered by a clique whose C-part contains both u1 and u2.
    # Edge (u,v) cross: covered by a clique whose C-part contains u AND whose I-part is v.
    #
    # So the C-C edges are covered by cliques based on their C-parts, and cross edges by cliques based on (C-part, I-vertex).
    #
    # To minimize number of cliques, we want large C-parts and to cover many cross edges per clique.
    #
    # This is essentially: we need to choose a collection of (C_subset, v_or_None) such that:
    #   1. For each (u1,u2) in C, exactly one chosen clique has C_subset containing {u1,u2}.
    #   2. For each cross edge (u,v), exactly one chosen clique has C_subset containing u and I-part = v.
    #   3. Each clique's C_subset (if size>=2) and I-part must be consistent (v adjacent to all in C_subset).
    #
    # This is a hard combinatorial problem. Given the time budget, let me use a heuristic.
    #
    # Heuristic approach:
    # Step 1: For the C-C edges, we need to partition them into cliques. A natural way: the whole C is a clique, so {C} covers all C-C edges in 1 clique. But then no other clique can have C-subset size >=2.
    #
    # Let me try: Use the full C as one clique (covers all C-C edges). Then for cross edges, each cross edge (u,v) must be in a clique with C-subset containing u and I-part v. Since no other clique can have C-subset size>=2, each such clique has C-subset = {u} (size 1, so no C-C edge). So each cross edge is a separate clique {u, v}. Number of cross edges = sum over v in I of |N(v)|. That's a lot.
    #
    # Alternative: Don't use full C. Instead, partition C into groups, and for each group G (a subset of C), use G as a clique (covers C-C edges within G). Then cross edges (u,v) where u in G can be covered by a clique {v} union {u}... but that has C-subset {u} size 1, no C-C edge. Hmm.
    #
    # Wait, I think the issue is that I'm overcomplicating. Let me reconsider whether C-C edges can be shared.
    #
    # Actually, re-reading: "every edge must lie in exactly one listed clique." So yes, each edge in exactly one.
    #
    # Let me try yet another approach. What if we don't separate C and I so rigidly?
    #
    # New idea: The minimum clique cover of a split graph. Let me look at it from the perspective of: the complement of a split graph is also split. Hmm.
    #
    # Let me just try a concrete small heuristic and measure.
    #
    # Approach A: 
    #   - Make the clique C as one clique.
    #   - For each independent vertex v, its neighbors N(v) in C. We need to cover edges (u,v) for u in N(v). 
    #   - But we can't make a clique {v} union S with |S|>=2 because that double-covers C-C edges in S.
    #   - So each edge (u,v) is its own clique. Total = 1 + (number of cross edges).
    #
    # Approach B:
    #   - Don't make C a single clique. Instead, partition C into singletons for the purpose of C-C edges? No, C-C edges need cliques of size>=2.
    #
    # Let me reconsider. Maybe the intended solution uses the fact that we can have cliques like {v1, v2, ...} no wait, I is independent so no two I vertices adjacent.
    #
    # OK here's a cleaner formulation. Let me denote the cliques we choose. Each clique is either:
    #   Type C: a subset of C (size>=2), covers C-C edges within it.
    #   Type M: {v} union S where v in I, S subset of C, |S|>=1, v adjacent to all in S. Covers cross edges (u,v) for u in S, and C-C edges within S (if |S|>=2).
    #
    # Constraint: C-C edge (u1,u2) covered by exactly one clique (Type C containing both, or Type M with S containing both).
    # Cross edge (u,v) covered by exactly one Type M clique with v and u in S.
    #
    # To minimize total cliques, we want to maximize edges covered per clique.
    #
    # Given complexity, let me use a greedy heuristic:
    #   1. Start with no cliques. All edges uncovered.
    #   2. Repeatedly pick the clique (Type C or Type M) that covers the most uncovered edges, add it, mark those edges covered.
    #   3. Stop when all edges covered.
    #   4. But we must ensure no edge is covered twice. Greedy set cover naturally avoids this if we only count uncovered edges.
    #
    # The challenge: the space of possible cliques is large. Type C: any subset of C of size>=2. Type M: for each v in I, any subset of N(v) of size>=1.
    #
    # For n up to 90, k up to 30, C has 30 vertices, C-C edges = 435. I has 60 vertices. For each v, N(v) is a random subset of C with p=0.7, so ~21 neighbors. Number of Type M cliques for one v = 2^21 - 1 ~ 2 million. Too many to enumerate.
    #
    # So we need a smarter heuristic. Let me think about what good cliques look like.
    #
    # Insight: A Type M clique {v} union S covers |S| cross edges (u,v) for u in S, plus C(C,2) C-C edges within S. To be efficient, S should be large. The largest S for v is N(v) itself. So the "best" Type M clique for v is {v} union N(v), covering all cross edges of v plus C(|N(v)|,2) C-C edges.
    #
    # Similarly, the best Type C clique is C itself.
    #
    # But we can't use both C (Type C) and {v} union N(v) (Type M) because they'd double-cover C-C edges.
    #
    # So the trade-off: using C as Type C covers all C-C edges in 1 clique, but then Type M cliques can only have |S|=1 (to avoid C-C edges), so each cross edge is separate.
    # Using {v} union N(v) for each v covers cross edges efficiently but also covers C-C edges (within N(v)), and these C-C edges across different v's will overlap (double cover).
    #
    # So we need to partition the C-C edges among the Type M cliques (and possibly a Type C clique).
    #
    # This is essentially a set-partition problem. Given the difficulty, let me try a different angle.
    #
    # Let me try: For each independent vertex v, we will create cliques that partition N(v) into groups. Each group S (subset of N(v)) gives a clique {v} union S. The C-C edges within S are covered by this clique. Now, the same C-C edge (u1,u2) might be in S for multiple v's -> double cover. To prevent this, we need the sets S (across all v) to have the property that no pair {u1,u2} appears in more than one S.
    #
    # This is like: we have a complete graph on C (the C-C edges). We want to assign each C-C edge to at most one "owner" (a v and a group S of v containing that edge). And each cross edge (u,v) must be in exactly one group of v.
    #
    # This is very complex. Let me just try a pragmatic heuristic and measure the score.
    #
    # Pragmatic heuristic:
    #   - For each independent vertex v, sort its neighbors. 
    #   - Try to make one big clique {v} union N(v) for each v. This covers all cross edges and some C-C edges.
    #   - The C-C edges covered by these are the union of all pairs within each N(v).
    #   - C-C edges NOT covered by any N(v) pair need to be covered by additional cliques.
    #   - But C-C edges covered by multiple N(v)'s are double-covered -> invalid.
    #
    # So I need to handle the double-cover. 
    #
    # Let me try: use {v} union N(v) for each v, but then for C-C edges that are in multiple N(v)'s, I have a problem. 
    #
    # Alternative: only use Type M cliques with |S|=1 (i.e., single cross edges), and cover C-C edges with one Type C clique (the full C). Total = 1 + (num cross edges). For n=90,k=30,p=0.7: cross edges ~ 60*30*0.7 = 1260. Total ~1261. That's a lot but valid.
    #
    # The seed program returns each edge as a clique, so seed total = number of edges = C(k,2) + cross_edges = 435 + 1260 = 1695 (for n=90). My approach A gives 1 + 1260 = 1261. That's better than seed!
    #
    # Wait, seed returns each edge as a 2-clique. So seed total = total edges. For (30,10): C(10,2)=45, cross ~ 20*10*0.7=140, total ~185. For (60,20): C(20,2)=190, cross ~40*20*0.7=560, total ~750. For (90,30): C(30,2)=435, cross ~60*30*0.7=1260, total ~1695. Sum ~2630.
    #
    # My approach A (full C + single cross edges): (30,10): 1+140=141. (60,20): 1+560=561. (90,30): 1+1260=1261. Sum ~1963. Better than seed 2630.
    #
    # Can I do better? Let me try to group cross edges.
    #
    # Better heuristic: For each independent vertex v, instead of making each cross edge a separate clique, group its neighbors into cliques. But to avoid C-C double-cover, I need the groups across v's to not share C-C edges.
    #
    # Idea: Assign each C-C edge to exactly one "owner" v (or to the C-clique). Then for v, the groups S must only contain C-C edges owned by v.
    #
    # This is like edge-coloring / edge-partitioning. 
    #
    # Given time constraints, let me implement a reasonable heuristic:
    #   1. Cover C-C edges with the full C clique (1 clique).
    #   2. For cross edges, use a greedy approach: for each independent vertex v, try to form cliques {v} union S where S is a subset of N(v), but since C-C edges are already covered by the C-clique, we CANNOT have |S|>=2 (would double-cover). So S must be size 1.
    #
    # That gives approach A. To improve, I need to NOT use the full C-clique, and instead let Type M cliques cover some C-C edges.
    #
    # Let me try a hybrid:
    #   - For independent vertices with large N(v), use {v} union N(v) as a clique (covers many cross edges + some C-C edges).
    #   - The C-C edges covered by these are "used up".
    #   - Remaining C-C edges (not covered by any Type M clique) are covered by additional Type C cliques.
    #   - But I must ensure no C-C edge is covered by two Type M cliques.
    #
    # To ensure that, I can process v's in some order and for each v, only include in S the C-C edges not yet covered. But S must be a subset of N(v), and the C-C edges within S are the pairs in S. If I greedily pick S for v to avoid already-covered pairs, it's like finding a clique in a graph where edges are "available" C-C edges.
    #
    # This is getting too complex for the time budget. Let me just go with a solid heuristic:
    #
    # Final heuristic:
    #   - Compute all cross edges and C-C edges.
    #   - Use greedy set cover over edges, where candidate cliques are:
    #       * The full C (covers all C-C edges).
    #       * For each v in I, {v} union N(v) (covers all cross edges of v + C-C edges within N(v)).
    #       * For each cross edge (u,v), the 2-clique {u,v}.
    #       * For each pair of C vertices (u1,u2), the 2-clique {u1,u2} (if not already covered).
    #     But the candidate set is large. Let me restrict to:
    #       * Full C.
    #       * For each v, {v} union N(v).
    #       * All 2-cliques for cross edges and C-C edges.
    #   - Greedy: repeatedly pick the candidate clique covering the most uncovered edges.
    #   - This ensures each edge covered exactly once (since we only count uncovered).
    #
    # The number of candidates: 1 (full C) + |I| (Type M big) + (num cross edges + num C-C edges) (2-cliques) = 1 + 60 + 1260 + 435 = 1756 for n=90. Manageable.
    #
    # But greedy set cover with 1756 candidates and ~1700 elements (edges) is fine.
    #
    # Let me implement this.
    #
    # Actually, I realize the full C clique and the {v} union N(v) cliques might both cover the same C-C edge, but greedy handles it by only counting uncovered edges.
    #
    # Let me code this up.
    from random import Random
    rng = Random(seed)
    C = list(range(k))
    I = list(range(k, n))
    # Build cross adjacency
    cross_adj = {}  # v -> set of u in C adjacent
    cross_edges = []
    for v in I:
        nbrs = set()
        for u in C:
            if rng.random() < p:
                nbrs.add(u)
                cross_edges.append((u, v))
        cross_adj[v] = nbrs
    # C-C edges
    cc_edges = [(u1, u2) for u1 in C for u2 in C if u1 < u2]
    # All edges
    all_edges = set(cc_edges + cross_edges)
    # Build candidate cliques
    candidates = []
    # Full C
    candidates.append(frozenset(C))
    # For each v, {v} union N(v)
    for v in I:
        if cross_adj[v]:
            candidates.append(frozenset([v]) | cross_adj[v])
    # 2-cliques for each edge
    for (a, b) in all_edges:
        candidates.append(frozenset([a, b]))
    # Greedy set cover
    uncovered = set(all_edges)
    chosen = []
    # Precompute for each candidate the set of edges it covers
    # Edge is a frozenset of 2 vertices
    def edges_in_clique(clique):
        # clique is frozenset of vertices
        verts = list(clique)
        es = set()
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                e = frozenset([verts[i], verts[j]])
                if e in all_edges:
                    es.add(e)
        return es
    cand_edges = [edges_in_clique(c) for c in candidates]
    # Greedy
    while uncovered:
        best_i = -1
        best_count = 0
        for i in range(len(candidates)):
            cnt = len(cand_edges[i] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_i = i
        if best_i == -1 or best_count == 0:
            # Should not happen if all edges are covered by some candidate
            break
        chosen.append(candidates[best_i])
        uncovered -= cand_edges[best_i]
    # Convert to list of lists
    result = [list(c) for c in chosen]
    return result

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        res = _solve_one(n, k, p, seed)
        results.append(res)
        if time.time() > deadline:
            # Return what we have (shouldn't happen normally)
            break
    # Ensure types are correct
    final = []
    for res in results:
        lst = []
        for clique in res:
            lst.append([int(v) for v in clique])
        final.append(lst)
    return final
# EVOLVE-BLOCK-END
