import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_triangles(adj, n):
        """Find all triangles, return list of (u,v,w) with u<v<w."""
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))
        return triangles

    def extend_clique(clique, adj, remaining, n, deadline, rng):
        """Extend a clique greedily by adding vertices with best connectivity."""
        if time.time() > deadline:
            return clique

        candidates = set.intersection(*(adj[v] for v in clique)) - set(clique)
        if not candidates:
            return clique

        while candidates and time.time() < deadline:
            best = None
            best_score = -1
            for c in candidates:
                # Check all edges from c to clique are in remaining
                all_valid = True
                for v in clique:
                    e = (min(c, v), max(c, v))
                    if e not in remaining:
                        all_valid = False
                        break
                if not all_valid:
                    continue

                # Score: number of remaining edges from c to other candidates
                s = 0
                for other in candidates:
                    if other != c:
                        e = (min(c, other), max(c, other))
                        if e in remaining:
                            s += 1
                if s > best_score:
                    best_score = s
                    best = c

            if best is None:
                break

            clique.add(best)
            candidates = candidates & adj[best]
            candidates -= clique

        return clique

    def greedy_partition(edges, n, adj, deadline, rng):
        """One greedy pass: seed with triangles, extend, cover rest as pairs."""
        remaining = set(edges)
        cliques = []

        if not remaining:
            return cliques

        # Find triangles and score them by extension potential
        triangles = find_triangles(adj, n)
        tri_scores = []
        for (u, v, w) in triangles:
            e1 = (u, v)
            e2 = (u, w)
            e3 = (v, w)
            if e1 in remaining and e2 in remaining and e3 in remaining:
                common = adj[u] & adj[v] & adj[w]
                # Count how many of those common neighbors still have all edges in remaining
                ext_count = 0
                for x in common:
                    ok = True
                    for y in (u, v, w):
                        e = (min(x, y), max(x, y))
                        if e not in remaining:
                            ok = False
                            break
                    if ok:
                        ext_count += 1
                tri_scores.append((ext_count, u, v, w))

        tri_scores.sort(reverse=True)

        # Greedily pick and extend triangles
        for score, u, v, w in tri_scores:
            if time.time() > deadline:
                break
            e1 = (u, v)
            e2 = (u, w)
            e3 = (v, w)
            if e1 not in remaining or e2 not in remaining or e3 not in remaining:
                continue

            clique = {u, v, w}
            clique = extend_clique(clique, adj, remaining, n, deadline, rng)

            if len(clique) >= 2:
                clique_list = sorted(clique)
                # Verify all edges are in remaining
                valid = True
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                        if e not in remaining:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for i in range(len(clique_list)):
                        for j in range(i+1, len(clique_list)):
                            e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                            remaining.discard(e)
                    cliques.append(clique_list)

        # Second pass: try to form cliques from remaining edges using degree ordering
        remaining_adj = [set() for _ in range(n)]
        for u, v in remaining:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)

        deg_order = sorted(range(n), key=lambda x: len(remaining_adj[x]), reverse=True)

        for u in deg_order:
            if time.time() > deadline:
                break
            if not remaining_adj[u]:
                continue

            clique = {u}
            candidates = set(remaining_adj[u])

            while candidates and time.time() < deadline:
                best = None
                best_score = -1
                for c in candidates:
                    all_valid = True
                    for v in clique:
                        e = (min(c, v), max(c, v))
                        if e not in remaining:
                            all_valid = False
                            break
                    if not all_valid:
                        continue
                    s = len(clique & remaining_adj[c])
                    # Secondary: edges to other candidates
                    s2 = sum(1 for other in candidates if other != c and (min(c, other), max(c, other)) in remaining)
                    total = s * 100 + s2
                    if total > best_score:
                        best_score = total
                        best = c

                if best is None:
                    break

                clique.add(best)
                candidates = candidates & remaining_adj[best]
                candidates -= clique

            if len(clique) >= 2:
                clique_list = sorted(clique)
                valid = True
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                        if e not in remaining:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for i in range(len(clique_list)):
                        for j in range(i+1, len(clique_list)):
                            e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                            remaining.discard(e)
                    cliques.append(clique_list)

        # Cover remaining edges as pairs
        for edge in list(remaining):
            if time.time() > deadline:
                break
            u, v = edge
            if (u, v) in remaining:
                cliques.append([u, v])
                remaining.discard((u, v))

        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)

        best_cliques = None
        best_count = float('inf')
        end_time = time.time() + time_budget

        # Try multiple random restarts with different orderings
        restart_num = 0
        while time.time() < end_time:
            rng = random.Random(seed * 1000 + restart_num)
            cliques = greedy_partition(edges, n, adj, end_time, rng)

            # Verify completeness
            covered = set()
            for c in cliques:
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        covered.add(e)
            if covered == edges:
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques

            restart_num += 1
            if time.time() > end_time - 3:
                break

        if best_cliques is None:
            # Fallback: each edge as a clique
            best_cliques = [list(e) for e in edges]

        return best_cliques

    # Solve both instances
    t1 = time.time()
    budget1 = min(55, DEADLINE - (t1 - start_time) - 15)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(55, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
