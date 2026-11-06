"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
            
            # Find multiple candidate cliques and pick the best one
            # Best = covers most remaining edges, with tie-break on size
            best_clique = None
            best_score = -1
            
            # Try max clique from different starting points
            candidates = []
            
            # Candidate 1: max clique from all active vertices
            C1 = max_clique(active, rem_adj, rng)
            if len(C1) >= 2:
                candidates.append(C1)
            
            # Candidate 2: max clique starting from highest-degree vertex
            if active:
                max_deg_vertex = max(active, key=lambda v: len(rem_adj[v]))
                C2 = max_clique({max_deg_vertex} | rem_adj[max_deg_vertex] & active, rem_adj, rng)
                if len(C2) >= 2 and C2 != C1:
                    candidates.append(C2)
            
            # Candidate 3: Try a random subset of active vertices
            if len(active) > 5:
                sample_size = min(8, len(active))
                sample = list(rng.sample(list(active), sample_size))
                C3 = max_clique(set(sample), rem_adj, rng)
                if len(C3) >= 2 and C3 not in candidates:
                    candidates.append(C3)
            
            # Score each candidate: count how many remaining edges it covers
            for C in candidates:
                edge_count = 0
                C_set = set(C)
                for u in C:
                    for v in C_set:
                        if u < v:
                            e = (u, v)
                            if e in remaining:
                                edge_count += 1
                # Score: primarily by edge coverage, secondarily by clique size
                score = edge_count * 100 + len(C)
                if score > best_score:
                    best_score = score
                    best_clique = C
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: just take one edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue
            
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
        changed = True
        while changed:
            changed = False
            T = len(cliques)
            for i in range(T):
                if changed:
                    break
                for j in range(i + 1, T):
                    Ci = set(cliques[i])
                    Cj = set(cliques[j])
                    # Check if union forms a clique
                    union = Ci | Cj
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u != v and v not in G_adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if not is_clique:
                        continue
                    # Check no other clique shares >1 vertex with union
                    ok2 = True
                    for k in range(T):
                        if k == i or k == j:
                            continue
                        if len(union & set(cliques[k])) > 1:
                            ok2 = False
                            break
                    if not ok2:
                        continue
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
        # Use more restarts since each greedy is now more expensive but potentially better
        max_attempts = 40
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
            if attempt >= max_attempts:
                break
        if best is None:
            best = [[u, v] for (u, v) in edges]
        return best

    result = []
    # first instance: n=32 (easier), then n=64 (harder, more budget)
    result.append(solve(32, 0.6, 601, t0 + 55.0))
    if time.time() > TOTAL_DEADLINE - 2.0:
        edges = gen_edges(64, 0.55, 602)
        result.append([[u, v] for (u, v) in edges])
    else:
        result.append(solve(64, 0.55, 602, TOTAL_DEADLINE - 1.0))
    return result
# EVOLVE-BLOCK-END
