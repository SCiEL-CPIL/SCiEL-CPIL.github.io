import time
import sys
from random import Random
from collections import Counter

_DEADLINE = time.time() + 140.0  # safety margin under 150s

def _elapsed():
    return time.time() - (_DEADLINE - 140.0)

def _deadline_hit():
    return time.time() > _DEADLINE

def construct():
    t0 = time.time()
    def deadline_hit():
        return (time.time() - t0) > 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Return list of maximal cliques (each a frozenset of vertices)."""
        cliques = []
        P = set(range(n))
        X = set()
        def bk(P, X):
            if not P and not X:
                cliques.append(frozenset(P) if P else None)  # placeholder
                # Actually we need the current clique; pass it as parameter
                return
            # Standard BK: pass R
        # Redo with R
        cliques.clear()
        def bk2(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            U = P | X
            if U:
                best_u = max(U, key=lambda u: len(P & adj[u]))
            else:
                best_u = None
            if best_u is not None:
                candidates = P - adj[best_u]
            else:
                candidates = set(P)
            for v in list(candidates):
                bk2(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk2(frozenset(), set(range(n)), set())
        return cliques

    def min_hitting_set(cliques, n):
        """Find minimum set of vertices hitting every clique (size>=2).
        Uses branch-and-bound with greedy upper bound and disjoint lower bound."""
        if not cliques:
            return []
        # Convert cliques to frozensets of ints
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            return []

        # Count frequency of each vertex
        freq = Counter()
        for c in cliques:
            for v in c:
                freq[v] += 1

        # Greedy upper bound
        remaining = [set(c) for c in cliques]
        covered = [False] * len(remaining)
        greedy_set = []
        while True:
            if all(covered):
                break
            # Pick vertex covering most uncovered cliques
            best_v = -1
            best_count = -1
            # Only consider vertices in uncovered cliques
            cand = set()
            for i, c in enumerate(remaining):
                if not covered[i]:
                    cand |= c
            for v in cand:
                cnt = 0
                for i, c in enumerate(remaining):
                    if not covered[i] and v in c:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v < 0:
                break
            greedy_set.append(best_v)
            for i, c in enumerate(remaining):
                if not covered[i] and best_v in c:
                    covered[i] = True
        ub = len(greedy_set)

        # Lower bound: pack disjoint cliques greedily
        def lower_bound():
            lb = 0
            used = [False] * len(remaining)
            for i, c in enumerate(remaining):
                if not used[i]:
                    lb += 1
                    for j in range(i + 1, len(remaining)):
                        if not used[j] and (c & remaining[j]):
                            used[j] = True
            return lb

        lb_init = lower_bound()

        # Order cliques by size (smaller first for tighter branching)
        order = sorted(range(len(remaining)), key=lambda i: len(remaining[i]))
        # Order vertices by frequency descending
        vert_order = sorted(freq.keys(), key=lambda v: -freq[v])

        best = list(greedy_set)
        best_size = ub
        current = []

        # Branch-and-bound
        # State: index in ordered cliques, current set
        sys.setrecursionlimit(10000)

        def solve(idx, current_set):
            nonlocal best, best_size
            if deadline_hit():
                return
            if len(current_set) >= best_size:
                return
            if idx == len(order):
                if len(current_set) < best_size:
                    best = list(current_set)
                    best_size = len(current_set)
                return
            ci = order[idx]
            c = remaining[ci]
            # Lower bound: current size + (number of remaining cliques that are disjoint from each other and not hit)
            # Simple LB: count remaining cliques that are not hit by current_set
            rem_unhit = 0
            for j in range(idx, len(order)):
                cj = remaining[order[j]]
                if not (cj & set(current_set)):
                    rem_unhit += 1
            # A single vertex can hit at most all unhit, so LB = len(current_set) + 1 if rem_unhit > 0
            if len(current_set) + (1 if rem_unhit > 0 else 0) >= best_size:
                return
            # If this clique is already hit, move on
            if c & set(current_set):
                solve(idx + 1, current_set)
                return
            # Branch: pick a vertex from c
            # Order vertices in c by frequency descending
            for v in sorted(c, key=lambda v: -freq[v]):
                if len(current_set) + 1 >= best_size:
                    break
                current_set.append(v)
                solve(idx + 1, current_set)
                current_set.pop()
                if deadline_hit():
                    return

        solve(0, [])

        # Local polish: try removing any vertex
        best_set = set(best)
        changed = True
        while changed and not deadline_hit():
            changed = False
            for v in list(best_set):
                trial = best_set - {v}
                ok = True
                for c in remaining:
                    if not (c & trial):
                        ok = False
                        break
                if ok:
                    best_set = trial
                    changed = True
        return sorted(best_set)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch(adj, n)
        # Filter to size >= 2
        cliques = [c for c in cliques if c and len(c) >= 2]
        hs = min_hitting_set(cliques, n)
        results.append(hs)
        if deadline_hit():
            # If deadline hit, use greedy for remaining
            for n2, p2, seed2 in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
                if (n2, p2, seed2) not in [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)][:len(results)]:
                    pass
            break
    # Ensure we have 3 results
    while len(results) < 3:
        n, p, seed = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results)]
        edges = instance(n, p, seed)
        touched = {x for e in edges for x in e}
        results.append(sorted(touched))
    return results
