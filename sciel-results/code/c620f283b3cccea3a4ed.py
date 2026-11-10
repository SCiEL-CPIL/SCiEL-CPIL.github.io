"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))

        # Build adjacency for cross-edges
        cross = {i: [] for i in I}
        for (a, b) in edges:
            if a < k and b < k:
                continue
            elif a >= k and b >= k:
                continue
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)

        # All C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        used_C_edges = set()

        # Phase 1: Cover cross-edges greedily.
        # For each I vertex, pair up its C-neighbors into 3-cliques {i, ca, cb}
        # using unused C-edges, then singles {i, c} for the rest.
        mixed_cliques = []
        for i in sorted(I, key=lambda x: -len(cross[x])):
            neighbors = cross[i]
            matched = set()
            for idx_a in range(len(neighbors)):
                if neighbors[idx_a] in matched:
                    continue
                c_a = neighbors[idx_a]
                for idx_b in range(idx_a + 1, len(neighbors)):
                    if neighbors[idx_b] in matched:
                        continue
                    c_b = neighbors[idx_b]
                    e = frozenset((c_a, c_b))
                    if e not in used_C_edges:
                        matched.add(c_a)
                        matched.add(c_b)
                        used_C_edges.add(e)
                        mixed_cliques.append([i, c_a, c_b])
                        break
            for c in neighbors:
                if c not in matched:
                    mixed_cliques.append([i, c])

        # Phase 2: Cover remaining C-edges with greedy maximal cliques.
        remaining = set(all_C_edges - used_C_edges)
        pure_cliques = []
        C_set = set(C)
        while remaining:
            if time.time() > deadline:
                break
            best_clique = None
            best_count = 0
            for start_v in C:
                # Build a maximal clique in the graph of remaining edges
                clique = [start_v]
                clique_set = {start_v}
                # candidates adjacent to all in clique via remaining edges
                candidates = [v for v in C if v != start_v
                              and all(frozenset((v, u)) in remaining for u in clique)]
                # Greedy: repeatedly add candidate that maximizes remaining edges within
                while candidates:
                    # pick candidate that maximizes number of remaining edges to other candidates+clique
                    def score(v):
                        s = 0
                        for u in clique:
                            if frozenset((v, u)) in remaining:
                                s += 1
                        for w in candidates:
                            if w != v and frozenset((v, w)) in remaining:
                                s += 1
                        return s
                    best_v = max(candidates, key=score)
                    clique.append(best_v)
                    clique_set.add(best_v)
                    candidates = [w for w in candidates if w != best_v
                                  and all(frozenset((w, u)) in remaining for u in clique_set)]
                # count remaining edges within clique
                count = sum(1 for e in remaining if e.issubset(clique_set))
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
            if best_count == 0 or best_clique is None:
                break
            pure_cliques.append(best_clique)
            for e in list(remaining):
                if e.issubset(set(best_clique)):
                    remaining.discard(e)

        # Phase 3: Local search refinement on pure cliques.
        # Try merging two pure cliques if union is a clique (all C-edges present).
        # Try splitting a pure clique into two smaller cliques to reduce count.
        def is_clique(vertices, edge_set):
            vs = set(vertices)
            for a in vertices:
                for b in vertices:
                    if a == b:
                        continue
                    if frozenset((a, b)) not in edge_set:
                        return False
            return True

        def all_C_edges_of(vertices):
            vs = list(vertices)
            return set(frozenset((a, b)) for i, a in enumerate(vs) for b in vs[i+1:])

        # Merging pass: repeatedly try to merge two cliques
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(pure_cliques)):
                for j in range(i + 1, len(pure_cliques)):
                    union = pure_cliques[i] + pure_cliques[j]
                    if is_clique(union, all_C_edges):
                        pure_cliques[i] = union
                        del pure_cliques[j]
                        changed = True
                        break
                if changed:
                    break

        # Splitting pass: try to split a clique into two smaller cliques
        # that together cover all edges of the original clique, reducing count.
        for idx in range(len(pure_cliques)):
            if time.time() > deadline:
                break
            cl = pure_cliques[idx]
            if len(cl) < 4:
                continue
            cl_edges = all_C_edges_of(cl)
            # Try to find a proper subset S of cl that is a clique and whose
            # complement (cl \ S) is also a clique, and the two cliques' edge sets
            # cover cl_edges. If so, splitting reduces count by 1.
            # Greedy: try S = cl minus one vertex, then minus two, etc.
            # Simpler: try all splits by removing a contiguous prefix.
            best_split = None
            for split in range(2, len(cl) - 1):
                S = cl[:split]
                T = cl[split:]
                if is_clique(S, all_C_edges) and is_clique(T, all_C_edges):
                    S_edges = all_C_edges_of(S)
                    T_edges = all_C_edges_of(T)
                    if S_edges | T_edges == cl_edges:
                        best_split = (S, T)
                        break
            if best_split:
                pure_cliques[idx] = best_split[0]
                pure_cliques.insert(idx + 1, best_split[1])

        result = mixed_cliques + pure_cliques
        # Ensure each clique is a list of ints, >= 2 distinct
        cleaned = []
        for cl in result:
            cl = list(cl)
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        return cleaned

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return edges as 2-cliques
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(solve(*spec))
    return results
# EVOLVE-BLOCK-END
