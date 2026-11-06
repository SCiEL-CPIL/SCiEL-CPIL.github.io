"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def color_sort(P, N, rng):
        order = []
        color = []
        uncolored = set(P)
        while uncolored:
            c = len(color) + 1
            candidates = list(uncolored)
            rng.shuffle(candidates)
            indep = []
            blocked = set()
            for v in candidates:
                if v in blocked or v not in uncolored:
                    continue
                indep.append(v)
                blocked |= N.get(v, set())
            for v in indep:
                uncolored.discard(v)
            for v in indep:
                order.append(v)
                color.append(c)
        return order, color

    def max_clique(active, rem_adj, rng):
        N = {u: (rem_adj[u] & active) for u in active}
        best = []
        def expand(R, P):
            nonlocal best
            if len(R) + len(P) <= len(best):
                return
            if not P:
                if len(R) > len(best):
                    best = R[:]
                return
            order, color = color_sort(P, N, rng)
            for i in range(len(order) - 1, -1, -1):
                if len(R) + color[i] <= len(best):
                    break
                v = order[i]
                if v in P:
                    expand(R + [v], P & N.get(v, set()))
                    P.discard(v)
        expand([], set(active))
        return best

    def count_edges_in_clique(C, rem_adj):
        """Count how many remaining edges are within clique C."""
        cnt = 0
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                if b in rem_adj[a]:
                    cnt += 1
        return cnt

    def greedy(n, edges_set, deadline, rng):
        remaining = set(edges_set)
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        cliques = []
        
        while remaining and time.time() < deadline - 1.5:
            active = set(u for u in range(n) if rem_adj[u])
            if not active:
                break
            
            # Find max clique
            C = max_clique(active, rem_adj, rng)
            
            if len(C) < 2:
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue
            
            # Try a few alternative cliques by perturbing the search
            best_clique = C
            best_score = len(C)
            
            # Try to find a clique that covers more edges relative to its size
            # by trying different random seeds for the coloring
            for trial in range(3):
                if time.time() > deadline - 2.0:
                    break
                alt_rng = Random(rng.random() * 1000000)
                alt_C = max_clique(active, rem_adj, alt_rng)
                if len(alt_C) > best_score:
                    best_score = len(alt_C)
                    best_clique = alt_C
                elif len(alt_C) == best_score:
                    # Prefer clique that covers more remaining edges
                    edges_alt = count_edges_in_clique(alt_C, rem_adj)
                    edges_best = count_edges_in_clique(best_clique, rem_adj)
                    if edges_alt > edges_best:
                        best_clique = alt_C
            
            C = best_clique
            
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    remaining.discard(e)
                    rem_adj[a].discard(b)
                    rem_adj[b].discard(a)
            cliques.append(C)
        
        for e in remaining:
            cliques.append([e[0], e[1]])
        return cliques

    def is_valid(partition, edges_set, G_adj):
        seen = set()
        for C in partition:
            if len(C) < 2:
                return False
            s = set(C)
            if len(s) != len(C):
                return False
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    if b not in G_adj[a]:
                        return False
                    e = (a, b) if a < b else (b, a)
                    if e in seen:
                        return False
                    seen.add(e)
        return seen == edges_set

    def merge_pass(cliques, G_adj):
        """Aggressive merge pass: try to merge pairs of cliques if their union is a clique
        and doesn't overlap too much with other cliques."""
        changed = True
        max_rounds = 50
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

    def solve(n, p, seed, budget):
        edges = gen_edges(n, p, seed)
        edges_set = set(edges)
        if not edges:
            return []
        G_adj = [set() for _ in range(n)]
        for u, v in edges:
            G_adj[u].add(v)
            G_adj[v].add(u)

        best = None
        base = seed * 100003 + 17
        attempt = 0
        while time.time() < budget:
            attempt += 1
            rng = Random(base + attempt)
            partition = greedy(n, edges_set, budget, rng)
            partition = merge_pass(partition, G_adj)
            if is_valid(partition, edges_set, G_adj):
                if best is None or len(partition) < len(best):
                    best = partition
                    if len(best) == 1:
                        break
            # Reduce attempts as we run out of time to ensure we finish
            if attempt >= 40:
                break
        if best is None:
            best = [[u, v] for (u, v) in edges]
        return best

    result = []
    # first instance: n=32 (easier), then n=64 (harder, more budget)
    # Give n=32 about 40s, n=64 the rest
    result.append(solve(32, 0.6, 601, t0 + 40.0))
    if time.time() > TOTAL_DEADLINE - 2.0:
        edges = gen_edges(64, 0.55, 602)
        result.append([[u, v] for (u, v) in edges])
    else:
        result.append(solve(64, 0.55, 602, TOTAL_DEADLINE - 1.0))
    return result
# EVOLVE-BLOCK-END
