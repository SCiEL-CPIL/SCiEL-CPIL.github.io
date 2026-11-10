def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))

        # Build cross adjacency: for each i in I, list of C-neighbors
        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].add(a)
            elif a >= k and b < k:
                cross[a].add(b)

        # All C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))

        # Greedy set cover for cross-edges
        # Universe: all cross-edges (i, c)
        # For each i in I, the possible cliques are {i} ∪ S where S ⊆ N(i)
        # A clique {i} ∪ S covers cross-edges (i, c) for c in S, AND C-edges within S
        # But we need "exactly one" coverage for ALL edges.
        
        # Strategy: 
        # 1. Cover cross-edges with mixed cliques {i} ∪ S_i
        # 2. The C-edges consumed by these mixed cliques are those within each S_i
        # 3. Remaining C-edges are covered by pure C-cliques
        # Constraint: no C-edge can be in two different cliques (mixed or pure)
        # So we need: the C-edges covered by mixed cliques must be disjoint, and 
        # the remaining C-edges (not covered by any mixed clique) are covered by pure cliques.
        
        # Actually, the constraint is: every edge lies in EXACTLY one listed clique.
        # Cross-edges (i,c): only can be in a mixed clique containing both i and c.
        #   So each cross-edge must be in exactly one mixed clique.
        # C-edges (u,v): can be in a pure C-clique OR in a mixed clique {i} ∪ S where u,v ∈ S.
        #   So each C-edge must be in exactly one clique (either pure or mixed).
        
        # Key insight: if a C-edge (u,v) is "used" by a mixed clique (both u,v in S_i),
        # then it cannot appear in any pure clique. Conversely, if not used by any mixed 
        # clique, it must be in exactly one pure clique.
        
        # So the optimization is:
        # - Choose for each i ∈ I a subset S_i ⊆ N(i) 
        # - Such that for any pair of C-vertices (u,v), at most one i has both u,v ∈ S_i
        #   (i.e., the sets {S_i : i ∈ I} have the property that no C-edge is in two of them)
        # - Minimize: (number of non-empty S_i) + (minimum clique cover of remaining C-edges)
        
        # Wait, but we also need each S_i to be a clique in the C-graph (which is complete, so any subset works).
        # And the cross-edges covered are (i, c) for c ∈ S_i. Every cross-edge (i,c) must be covered 
        # by exactly one mixed clique for vertex i. Since each i only appears in its own mixed cliques,
        # we need S_i to be exactly N(i) to cover all cross-edges for i? No!
        
        # Actually, cross-edge (i,c) can ONLY be in a clique that contains both i and c.
        # The only cliques containing i are mixed cliques {i} ∪ S where c ∈ S.
        # So for each i, the cross-edges (i,c) for c ∈ N(i) must be partitioned among the mixed cliques for i.
        # But wait - can i appear in multiple mixed cliques? Yes! The problem says "a list of cliques".
        # So i can be in multiple cliques {i} ∪ S_1, {i} ∪ S_2, etc., as long as:
        #   - Each cross-edge (i,c) is in exactly one of these (so S_1, S_2, ... partition N(i))
        #   - No C-edge is in two different cliques (mixed or pure)
        
        # This is more complex. Let me reconsider.
        
        # Simpler approach: for each i, use ONE mixed clique {i} ∪ N(i).
        # Then all cross-edges for i are covered. 
        # C-edges within N(i) are "consumed" by this clique.
        # Problem: if N(i) and N(j) share a C-edge (u,v), then (u,v) is in both clique_i and clique_j. INVALID.
        
        # So we need to ensure that for any pair (u,v) ∈ C, at most one i has both u,v ∈ N(i).
        # If conflicts exist, we must split some N(i) into smaller subsets.
        
        # Approach:
        # 1. Start with S_i = N(i) for all i.
        # 2. Find conflicts: pairs (i,j) where |S_i ∩ S_j| ≥ 2.
        # 3. Resolve by splitting one of S_i or S_j.
        # 4. After resolving all conflicts, cover remaining C-edges with pure cliques.
        
        # Let me implement a greedy conflict resolution:
        
        S = {i: set(cross[i]) for i in I}
        
        # Find all conflicting pairs
        def find_conflicts(S):
            conflicts = []
            I_list = list(I)
            for idx_a in range(len(I_list)):
                for idx_b in range(idx_a + 1, len(I_list)):
                    i_a, i_b = I_list[idx_a], I_list[idx_b]
                    inter = S[i_a] & S[i_b]
                    if len(inter) >= 2:
                        conflicts.append((i_a, i_b, inter))
            return conflicts
        
        # Split a set S_i to resolve conflicts with others
        # We want to partition S_i into parts such that each part has intersection ≤ 1 with every other S_j
        def split_to_resolve(i, S, others):
            # others: list of (j, S_j) for j ≠ i
            S_i = S[i]
            if not S_i:
                return [set()]
            
            # Greedy: build parts one by one
            parts = []
            remaining = set(S_i)
            while remaining:
                part = set()
                # Try to add vertices from remaining
                # A vertex v can be added to part if for all j, |part ∪ {v} ∩ S_j| ≤ 1
                # i.e., for all j, if v ∈ S_j, then |part ∩ S_j| ≤ 0 (i.e., part ∩ S_j is empty or has size 0)
                # Wait: we need |part ∩ S_j| ≤ 1 for all j. When adding v:
                #   new intersection with S_j is (part ∩ S_j) ∪ ({v} if v ∈ S_j else {})
                #   So we need: for all j, if v ∈ S_j, then |part ∩ S_j| ≤ 0, else |part ∩ S_j| ≤ 1
                #   I.e., if v ∈ S_j, then part ∩ S_j must be empty.
                
                candidates = list(remaining)
                while candidates:
                    # Pick vertex that can be added
                    added = False
                    for idx, v in enumerate(candidates):
                        can_add = True
                        for (j, S_j) in others:
                            if v in S_j:
                                if part & S_j:
                                    can_add = False
                                    break
                        if can_add:
                            part.add(v)
                            remaining.discard(v)
                            candidates.remove(v)
                            added = True
                            break
                    if not added:
                        # No vertex can be added; start new part with the first remaining
                        v = candidates[0]
                        part.add(v)
                        remaining.discard(v)
                        candidates.remove(v)
                        break
                if part:
                    parts.append(part)
            return parts
        
        # Iteratively resolve conflicts
        max_iter = 1000
        for _ in range(max_iter):
            conflicts = find_conflicts(S)
            if not conflicts:
                break
            # Pick the first conflict
            i_a, i_b, inter = conflicts[0]
            # Split the one with larger intersection or larger set
            if len(S[i_a]) >= len(S[i_b]):
                i_to_split = i_a
                others = [(j, S[j]) for j in I if j != i_a and j != i_b]
                others.append((i_b, S[i_b]))
            else:
                i_to_split = i_b
                others = [(j, S[j]) for j in I if j != i_a and j != i_b]
                others.append((i_a, S[i_a]))
            
            new_parts = split_to_resolve(i_to_split, S, others)
            # Replace S[i_to_split] with the parts
            # But now i_to_split has multiple cliques
            # We need to track this differently
            
            # Let me restructure: maintain a list of cliques per I-vertex
            break
        
        # Let me restructure the whole approach:
        # Maintain cliques as a list. Each mixed clique is (i, S) where i ∈ I, S ⊆ C.
        # Start with one clique per i: (i, N(i))
        # Resolve conflicts by splitting.
        
        cliques = {i: [set(cross[i])] for i in I}
        
        for _ in range(2000):
            # Find a conflict: two cliques (i, S_i) and (j, S_j) with |S_i ∩ S_j| ≥ 2
            conflict_found = False
            I_list = list(I)
            for idx_a in range(len(I_list)):
                if conflict_found:
                    break
                i_a = I_list[idx_a]
                for idx_b in range(idx_a + 1, len(I_list)):
                    i_b = I_list[idx_b]
                    for S_a in cliques[i_a]:
                        for S_b in cliques[i_b]:
                            inter = S_a & S_b
                            if len(inter) >= 2:
                                # Conflict found. Split the larger one.
                                if len(S_a) >= len(S_b):
                                    i_to_split = i_a
                                    S_to_split = S_a
                                    others = []
                                    for j in I:
                                        if j == i_a:
                                            for S_j in cliques[i_a]:
                                                if S_j is not S_a:
                                                    others.append((j, S_j))
                                        elif j == i_b:
                                            for S_j in cliques[i_b]:
                                                others.append((j, S_j))
                                        else:
                                            for S_j in cliques[j]:
                                                others.append((j, S_j))
                                else:
                                    i_to_split = i_b
                                    S_to_split = S_b
                                    others = []
                                    for j in I:
                                        if j == i_b:
                                            for S_j in cliques[i_b]:
                                                if S_j is not S_b:
                                                    others.append((j, S_j))
                                        elif j == i_a:
                                            for S_j in cliques[i_a]:
                                                others.append((j, S_j))
                                        else:
                                            for S_j in cliques[j]:
                                                others.append((j, S_j))
                                
                                # Split S_to_split
                                new_parts = []
                                remaining = set(S_to_split)
                                while remaining:
                                    part = set()
                                    candidates = list(remaining)
                                    while candidates:
                                        added = False
                                        for idx, v in enumerate(candidates):
                                            can_add = True
                                            for (j, S_j) in others:
                                                if v in S_j and (part & S_j):
                                                    can_add = False
                                                    break
                                            if can_add:
                                                part.add(v)
                                                remaining.discard(v)
                                                candidates.remove(v)
                                                added = True
                                                break
                                        if not added:
                                            v = candidates[0]
                                            part.add(v)
                                            remaining.discard(v)
                                            candidates.remove(v)
                                            break
                                    if part:
                                        new_parts.append(part)
                                
                                # Replace
                                if i_to_split == i_a:
                                    cliques[i_a].remove(S_a)
                                else:
                                    cliques[i_b].remove(S_b)
                                cliques[i_to_split].extend(new_parts)
                                conflict_found = True
                                break
                            if conflict_found:
                                break
                        if conflict_found:
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
            if not conflict_found:
                break
        
        # Now build the mixed cliques and track consumed C-edges
        mixed_cliques = []
        consumed_C_edges = set()
        
        for i in I:
            for S in cliques[i]:
                if len(S) < 2:
                    # Clique {i} ∪ S has size 1 + len(S). Need at least 2 vertices total.
                    # If S is empty, clique is {i} which has 1 vertex - invalid.
                    # If S has 1 element, clique is {i, c} which has 2 vertices - valid.
                    if len(S) == 0:
                        # Cannot form a valid clique. But we need to cover cross-edges for i.
                        # This shouldn't happen if cross[i] is non-empty.
                        continue
                    else:
                        mixed_cliques.append([i] + list(S))
                        # C-edges within S: only if len(S) >= 2
                        if len(S) >= 2:
                            S_list = list(S)
                            for idx_u in range(len(S_list)):
                                for idx_v in range(idx_u + 1, len(S_list)):
                                    consumed_C_edges.add(frozenset((S_list[idx_u], S_list[idx_v])))
                else:
                    mixed_cliques.append([i] + list(S))
                    S_list = list(S)
                    for idx_u in range(len(S_list)):
                        for idx_v in range(idx_u + 1, len(S_list)):
                            consumed_C_edges.add(frozenset((S_list[idx_u], S_list[idx_v])))
        
        # Remaining C-edges
        remaining_C_edges = all_C_edges - consumed_C_edges
        
        # Cover remaining C-edges with pure C-cliques
        # Greedy: repeatedly find the clique in C that covers the most remaining edges
        pure_cliques = []
        remaining = set(remaining_C_edges)
        
        while remaining:
            best_clique = None
            best_count = 0
            for start in C:
                clique = [start]
                clique_set = {start}
                # Find maximal clique in the graph of remaining edges
                # A vertex v can be added if (v, u) is in remaining for all u in clique
                candidates = [v for v in C if v != start]
                while candidates:
                    # Pick candidate with most connections in remaining within candidates ∪ clique
                    best_cand = None
                    best_deg = -1
                    for v in candidates:
                        deg = 0
                        for u in clique_set:
                            if frozenset((v, u)) in remaining:
                                deg += 1
                        if deg == len(clique_set):
                            # v is adjacent to all in clique
                            # Count edges to other candidates
                            cand_deg = 0
                            for w in candidates:
                                if w != v and frozenset((v, w)) in remaining:
                                    cand_deg += 1
                            if cand_deg > best_deg:
                                best_deg = cand_deg
                                best_cand = v
                    if best_cand is None:
                        break
                    clique.append(best_cand)
                    clique_set.add(best_cand)
                    candidates.remove(best_cand)
                
                # Count remaining edges within clique
                count = 0
                for e in remaining:
                    if e.issubset(clique_set):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
            
            if best_clique is None or best_count == 0:
                break
            
            pure_cliques.append(best_clique)
            clique_set = set(best_clique)
            for e in list(remaining):
                if e.issubset(clique_set):
                    remaining.discard(e)
        
        result = mixed_cliques + pure_cliques
        
        # Convert to lists of ints
        final_result = []
        for clique in result:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
