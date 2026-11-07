"""Lagrangian relaxation clique packing: dynamic edge weights guide greedy maximal clique selection."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    _T0 = time.time()
    _DEADLINE = 150.0 - 10.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _is_clique(adj, verts):
        vs = list(verts)
        for i in range(len(vs)):
            a = vs[i]
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[a]:
                    return False
        return True

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        # Split graph structure: C = [0, k), I = [k, n)
        # C is a clique, I is independent
        # Each clique is either subset of C, or {v} ∪ S where v in I, S subset of N_C(v), and S is clique in C (always true since C is clique)
        # So for each v in I, the max clique containing v is {v} ∪ N_C(v)
        
        cliques = []
        
        # Start with the full clique C if it has edges
        if k >= 2:
            C = list(range(k))
            cliques.append(C)
        
        # For each vertex in I, create a clique {v} ∪ N_C(v)
        for v in range(k, n):
            neighbors_in_C = [u for u in range(k) if u in adj[v]]
            if neighbors_in_C:
                clique = [v] + neighbors_in_C
                cliques.append(sorted(clique))
        
        # Validation
        for cl in cliques:
            assert len(cl) >= 2
            assert len(set(cl)) == len(cl)
            assert _is_clique(adj, cl)
        
        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (cl[a], cl[b])
                    assert e not in seen, "edge in two cliques"
                    seen.add(e)
        
        assert seen == all_edges, f"edge not covered: {len(all_edges - seen)} edges missing"
        
        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
