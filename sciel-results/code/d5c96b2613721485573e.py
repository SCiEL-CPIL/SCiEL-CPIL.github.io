"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = 125.0  # explicit guard, safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # adjacency sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(edges)
        cliques = []
        C = set(range(k))
        I = set(range(k, n))

        def remove_clique_edges(clique):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    remaining.discard((a, b) if a < b else (b, a))

        # ---- Step 1: cover cross edges ----
        # For each independent vertex w, N_C(w) = {u in C : (u,w) in E}
        # We must cover all cross edges. Each clique containing w can include
        # any subset of N_C(w) plus optional internal vertices.
        # Strategy: process independent vertices greedily. For each w, form
        # clique = {w} ∪ S where S ⊆ N_C(w) ∪ (optional internal vertices).
        # The clique must be a valid clique: all pairs in S must be edges.
        # Since C is a clique, any subset of C is a clique. w connects to N_C(w).
        # So {w} ∪ S is a clique iff S ⊆ N_C(w) ∪ {v : v not in C and v-w edge}
        # But we only use internal vertices from C to keep it simple.
        # Actually {w} ∪ S where S ⊆ N_C(w) is always a clique since C is a clique
        # and w is connected to all of N_C(w).
        #
        # To maximize internal edge coverage: for each w, the clique {w} ∪ N_C(w)
        # covers all internal edges among N_C(w). We want to order vertices so
        # that high-overlap neighborhoods are processed in an order that
        # maximizes total unique internal edges covered.
        #
        # Greedy: repeatedly pick the w that maximizes the number of internal
        # edges among N_C(w) that are still in `remaining`.
        
        # Precompute N_C(w) for each w in I
        N_C = {}
        for w in I:
            N_C[w] = set(adj[w] & C)

        pending = list(I)
        # Remove vertices with no cross edges (they don't need a clique)
        pending = [w for w in pending if N_C[w]]

        # Sort by degree in C descending as initial heuristic
        pending.sort(key=lambda w: len(N_C[w]), reverse=True)

        while pending:
            if time.time() > deadline:
                break
            # Pick the w that covers the most remaining internal edges
            best_w = None
            best_count = -1
            for w in pending:
                # Count internal edges among N_C(w) that are still remaining
                nc = N_C[w]
                cnt = 0
                for u in nc:
                    for v in nc:
                        if u < v and (u, v) in remaining:
                            cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            if best_w is None:
                break
            pending.remove(best_w)
            nc = N_C[best_w]
            clique = list(nc) + [best_w]
            cliques.append(clique)
            remove_clique_edges(clique)

        # ---- Step 2: cover remaining edges within the clique part ----
        # Repeatedly find a maximum clique in the residual graph
        while remaining:
            if time.time() > deadline:
                break
            # Build residual adjacency
            active = set()
            for u, v in remaining:
                active.add(u)
                active.add(v)
            radj = {v: set() for v in active}
            for u, v in remaining:
                radj[u].add(v)
                radj[v].add(u)
            
            # Find a large clique greedily (Bron-Kerbosch for max clique)
            best = []
            best_size = 0
            
            def bk(R, P, X):
                nonlocal best, best_size
                if not P and not X:
                    if len(R) > best_size:
                        best_size = len(R)
                        best = list(R)
                    return
                union = P | X
                if union:
                    pivot = max(union, key=lambda u: len(radj[u] & P))
                    candidates = P - radj[pivot]
                else:
                    candidates = set(P)
                for v in list(candidates):
                    bk(R | {v}, P & radj[v], X & radj[v])
                    P.remove(v)
                    X.add(v)
            
            bk(set(), set(active), set())
            
            if len(best) < 2:
                e = next(iter(remaining))
                best = [e[0], e[1]]
            
            cliques.append(best)
            remove_clique_edges(best)

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            # fallback: individual edges (valid, each size 2)
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)

    return results
# EVOLVE-BLOCK-END
