import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        remaining = set(edges)
        cliques = []

        def count_bits(x):
            return bin(x).count('1')

        def greedy_clique_from(start, remaining):
            """Build a clique starting from 'start' using only remaining edges."""
            clique = [start]
            # Candidates must be connected to start via a remaining edge
            cand_set = set()
            for u, v in remaining:
                if u == start:
                    cand_set.add(v)
                elif v == start:
                    cand_set.add(u)

            while True:
                if not cand_set:
                    break
                # For each candidate, check if connected to all in clique via remaining edges
                valid = []
                for cand in cand_set:
                    ok = True
                    for member in clique:
                        if member == cand:
                            continue
                        e = (min(member, cand), max(member, cand))
                        if e not in remaining:
                            ok = False
                            break
                    if ok:
                        valid.append(cand)
                if not valid:
                    break
                # Pick the candidate with the most remaining edges to other valid candidates
                best = max(valid, key=lambda c: sum(
                    1 for o in valid if o != c and (min(c, o), max(c, o)) in remaining
                ))
                clique.append(best)
                cand_set.discard(best)
                # Update candidates: must be connected to all clique members
                new_cand = set()
                for u, v in remaining:
                    for member in clique:
                        if u == member and v in cand_set:
                            # Check v connected to all other clique members
                            ok = True
                            for m2 in clique:
                                if m2 == v:
                                    continue
                                e2 = (min(m2, v), max(m2, v))
                                if e2 not in remaining:
                                    ok = False
                                    break
                            if ok:
                                new_cand.add(v)
                        elif v == member and u in cand_set:
                            ok = True
                            for m2 in clique:
                                if m2 == u:
                                    continue
                                e2 = (min(m2, u), max(m2, u))
                                if e2 not in remaining:
                                    ok = False
                                    break
                            if ok:
                                new_cand.add(u)
                cand_set = new_cand
            return clique if len(clique) >= 2 else None

        def find_best_clique(remaining):
            """Find the largest clique we can greedily construct from any start vertex."""
            best = None
            best_len = 0
            # Try vertices that still have remaining edges
            active = set()
            for u, v in remaining:
                active.add(u)
                active.add(v)

            for start in active:
                if time.time() > deadline - 5:
                    break
                c = greedy_clique_from(start, remaining)
                if c is None:
                    continue
                # Check that all edges in this clique are still remaining
                clique_set = set(c)
                valid = True
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        if e not in remaining:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                if len(c) > best_len:
                    best_len = len(c)
                    best = c
            return best

        def merge_pass(cliques, G_adj):
            """Try to merge pairs of cliques if their union is a clique."""
            changed = True
            max_rounds = 20
            round_count = 0
            while changed and round_count < max_rounds:
                changed = False
                round_count += 1
                T = len(cliques)
                for i in range(T):
                    if changed:
                        break
                    Ci = set(cliques[i])
                    for j in range(i + 1, T):
                        Cj = set(cliques[j])
                        # Union must be a clique
                        union = Ci | Cj
                        ok = True
                        for u in union:
                            for v in union:
                                if u == v:
                                    continue
                                if v not in G_adj[u]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # Check overlap with other cliques: union should share at most 1 vertex with each other clique
                        ok2 = True
                        for k in range(T):
                            if k == i or k == j:
                                continue
                            Ck = set(cliques[k])
                            if len(union & Ck) > 1:
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        # Merge: replace cliques[i] and cliques[j] with union
                        cliques[i] = list(union)
                        del cliques[j]
                        changed = True
                        break
            return cliques

        while remaining:
            if time.time() > deadline - 10:
                # Emergency: pack remaining edges as pairs
                rem_list = list(remaining)
                for e in rem_list:
                    cliques.append(list(e))
                break

            best = find_best_clique(remaining)
            if best is None:
                # Fallback: take one edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue

            # Remove all edges within the clique
            clique_set = set(best)
            to_remove = set()
            for u, v in remaining:
                if u in clique_set and v in clique_set:
                    to_remove.add((u, v))
            if to_remove:
                cliques.append(best)
                remaining.difference_update(to_remove)
            else:
                # Shouldn't happen, but fallback
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)

        # Apply merge pass to reduce clique count further
        G_adj = [set() for _ in range(n)]
        for u, v in edges:
            G_adj[u].add(v)
            G_adj[v].add(u)
        cliques = merge_pass(cliques, G_adj)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
