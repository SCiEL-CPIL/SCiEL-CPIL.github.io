"""Construct a large (5,5)-free graph via greedy vertex insertion + local search."""
import random
from itertools import combinations

def construct():
    random.seed(12345)
    CLIQUE = 5
    INDEPENDENT = 5
    MAX_VERTICES = 46  # upper bound proven; we try to reach high n

    def has_clique5(adj, n):
        # adj: list of sets
        for combo in combinations(range(n), 5):
            ok = True
            for a, b in combinations(combo, 2):
                if b not in adj[a]:
                    ok = False
                    break
            if ok:
                return True
        return False

    def has_indep5(adj, n):
        for combo in combinations(range(n), 5):
            ok = True
            for a, b in combinations(combo, 2):
                if b in adj[a]:
                    ok = False
                    break
            if ok:
                return True
        return False

    def is_free(adj, n):
        return not has_clique5(adj, n) and not has_indep5(adj, n)

    # Start from a small known-free seed: empty graph on 4 vertices is free.
    # We'll grow by adding vertices with a chosen adjacency pattern, then repair.
    # To be efficient, we build incrementally and verify at each step.

    # We'll attempt to build up to MAX_VERTICES. Use a heuristic: when adding
    # vertex v, choose its neighbor set to avoid creating a K5 or indep5 with
    # existing vertices. We try several random patterns and keep the first that
    # works; if none works at size n, we try local search on the last valid graph.

    def try_add_vertex(adj, n):
        # Try to add vertex n with various adjacency subsets to 0..n-1.
        # A vertex v creates a K5 if there exist 4 mutually-adjacent existing
        # vertices all adjacent to v (i.e., a K4 in N(v)). It creates an indep5
        # if there exist 4 mutually-non-adjacent existing vertices all non-adjacent
        # to v (i.e., an independent 4-set in non-N(v)).
        # We'll sample random subsets and test.
        existing = list(range(n))
        # Precompute: for each 4-subset, is it a clique? is it independent?
        cliques4 = set()
        indeps4 = set()
        for combo in combinations(existing, 4):
            allc = all(b in adj[a] for a, b in combinations(combo, 2))
            if allc:
                cliques4.add(frozenset(combo))
            alln = all(b not in adj[a] for a, b in combinations(combo, 2))
            if alln:
                indeps4.add(frozenset(combo))
        # A neighbor set S is safe iff: no clique4 is subset of S, and no indep4 is
        # subset of the complement (existing \ S).
        for attempt in range(20000):
            S = set(random.sample(existing, random.randint(0, n)))
            # check clique4 subset S
            bad = False
            for c4 in cliques4:
                if c4 <= S:
                    bad = True
                    break
            if bad:
                continue
            comp = set(existing) - S
            for i4 in indeps4:
                if i4 <= comp:
                    bad = True
                    break
            if bad:
                continue
            return S
        return None

    # Build up
    adj = [set() for _ in range(4)]
    # initialize 4 vertices with a random-free pattern (empty is safe)
    n = 4
    # try to grow
    max_reach = 4
    while n < MAX_VERTICES:
        S = try_add_vertex(adj, n)
        if S is None:
            # try a few more random restarts of the whole build below; break for now
            break
        # add vertex
        adj.append(set())
        for u in S:
            adj[n].add(u)
            adj[u].add(n)
        n += 1
        max_reach = n

    # Local search: try to add more vertices by also perturbing existing adjacencies?
    # Simpler: also try random restarts of the whole greedy build to find a larger n.
    best_n = n
    best_adj = [set(s) for s in adj]

    for restart in range(30):
        random.seed(1000 + restart)
        adj2 = [set() for _ in range(4)]
        n2 = 4
        while n2 < MAX_VERTICES:
            S = try_add_vertex(adj2, n2)
            if S is None:
                break
            adj2.append(set())
            for u in S:
                adj2[n2].add(u)
                adj2[u].add(n2)
            n2 += 1
        if n2 > best_n:
            best_n = n2
            best_adj = [set(s) for s in adj2]

    # Final verification
    if not is_free(best_adj, best_n):
        # fallback to guaranteed-free small graph
        best_n = 4
        best_adj = [set() for _ in range(4)]

    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))
    return best_n, edges
