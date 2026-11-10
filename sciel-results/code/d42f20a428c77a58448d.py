import time
import random
from collections import defaultdict

def construct():
    start = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    n = 15
    best_edges = None
    best_count = 0

    # Seed: star (14 edges, valid)
    def make_star():
        return [(0, v) for v in range(1, n)]

    # Greedy: try to add edges in random order
    def greedy_add(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = set()
        # Start with star
        for v in range(1, n):
            adj[0].add(v)
            adj[v].add(0)
            edges.add((0, v))

        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges]
        random.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if time.time() > DEADLINE:
                break
            # Try adding (u,v)
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))

            # Check validity: for every edge (a,b), N(a)\{b} and N(b)\{a} must have disjoint degree sets
            deg = [len(adj[x]) for x in range(n)]
            valid = True
            for (a, b) in list(edges):
                na = adj[a] - {b}
                nb = adj[b] - {a}
                degs_a = set(deg[x] for x in na)
                degs_b = set(deg[x] for x in nb)
                if degs_a & degs_b:
                    valid = False
                    break

            if not valid:
                # Revert
                adj[u].discard(v)
                adj[v].discard(u)
                edges.discard((u, v))

        return list(edges), len(edges)

    # Local search: remove edges to free up degree levels, then re-add
    def local_search(edges_list, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = set()
        for (u, v) in edges_list:
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))

        best_local = len(edges)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges]
            random.shuffle(all_pairs)

            # Try adding a random missing edge
            for (u, v) in all_pairs:
                if time.time() > DEADLINE:
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))

                deg = [len(adj[x]) for x in range(n)]
                valid = True
                for (a, b) in list(edges):
                    na = adj[a] - {b}
                    nb = adj[b] - {a}
                    degs_a = set(deg[x] for x in na)
                    degs_b = set(deg[x] for x in nb)
                    if degs_a & degs_b:
                        valid = False
                        break

                if valid:
                    best_local = len(edges)
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges.discard((u, v))

        return list(edges), len(edges)

    # Run multiple random restarts
    for seed in range(200):
        if time.time() > DEADLINE - 5:
            break
        edges_list, cnt = greedy_add(seed)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges_list

        # Local search polish
        if cnt > 14 and time.time() < DEADLINE - 3:
            edges_list2, cnt2 = local_search(edges_list, seed + 10000)
            if cnt2 > best_count:
                best_count = cnt2
                best_edges = edges_list2

    # If we never improved beyond star, return star
    if best_edges is None:
        best_edges = make_star()

    # Convert to proper format
    result = []
    for (u, v) in best_edges:
        result.append([int(u), int(v)])

    return result
