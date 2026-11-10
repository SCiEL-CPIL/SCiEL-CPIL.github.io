import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        if not edges:
            return []

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(edges)
        cliques = []
        rng = random.Random(12345)

        def extend_clique(start_clique, rem, adj):
            current = set(start_clique)
            # Find common neighbors
            if len(current) == 0:
                candidates = set(range(n))
            else:
                candidates = set.intersection(*[adj[v] for v in current])
                candidates -= current

            while candidates:
                # Score each candidate by number of edges to current in rem
                best_v = None
                best_score = -1
                for v in candidates:
                    score = 0
                    for u in current:
                        e = (min(u, v), max(u, v))
                        if e in rem:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v is None or best_score == 0:
                    break
                current.add(best_v)
                candidates = set.intersection(*[adj[v] for v in current])
                candidates -= current
            return current

        while remaining:
            if time.time() - start_time > time_limit:
                break

            # Pick random edge
            edge = rng.choice(list(remaining))
            u, v = edge

            clique = extend_clique({u, v}, remaining, adj)

            # Collect edges of clique that are in remaining
            clique_edges = set()
            cl = list(clique)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining:
                        clique_edges.add(e)

            if not clique_edges:
                clique_edges = {edge}

            remaining -= clique_edges
            cliques.append(cl)

        # Handle leftover edges
        if remaining:
            for e in remaining:
                cliques.append(list(e))

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        elapsed = time.time() - start_time
        time_left = DEADLINE - elapsed
        if time_left < 10:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
        else:
            cliques = solve_instance(n, p, seed, time_left)
            results.append(cliques)

    return results
