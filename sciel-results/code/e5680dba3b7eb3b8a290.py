import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_assign(edges, n, edge_order, deadline):
        """
        Assign each edge (in edge_order) to a clique.
        For each edge (u,v), try to add it to an existing clique C if C ∪ {u,v} is still a clique.
        Otherwise create a new clique {u,v}.
        """
        # Build adjacency as bitmask for fast clique checks
        adj = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        cliques = []  # list of [set, bitmask]

        for (u, v) in edge_order:
            if time.time() > deadline:
                break
            uv_mask = (1 << u) | (1 << v)

            best_clique_idx = -1
            best_size = 0

            # Try to fit into an existing clique
            for ci in range(len(cliques)):
                cset, cmask = cliques[ci]
                # Check if {u,v} can be added to cset
                if (cmask & ~adj[u]) == 0 and (cmask & ~adj[v]) == 0:
                    new_size = len(cset) + 2
                    if new_size > best_size:
                        best_size = new_size
                        best_clique_idx = ci

            if best_clique_idx >= 0:
                cset, cmask = cliques[best_clique_idx]
                cset.add(u)
                cset.add(v)
                cliques[best_clique_idx] = (cset, cmask | uv_mask)
            else:
                cliques.append(({u, v}, uv_mask))

        return [list(cset) for cset, cmask in cliques]

    def solve_instance(n, p, seed, deadline):
        edges = instance(n, p, seed)
        m = len(edges)

        # Compute degree for each vertex
        deg = [0] * n
        for (u, v) in edges:
            deg[u] += 1
            deg[v] += 1

        best_cliques = None
        best_count = float('inf')

        num_restarts = 80
        rng = random.Random(seed + 7)

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            # Create edge order: mix of degree-based and random
            # Use a weighted random shuffle
            if restart % 3 == 0:
                # Sort by degree product descending, then shuffle within ties
                order = sorted(edges, key=lambda e: (-deg[e[0]] * deg[e[1]], rng.random()))
            elif restart % 3 == 1:
                # Random shuffle
                order = list(edges)
                rng.shuffle(order)
            else:
                # Sort by min degree descending
                order = sorted(edges, key=lambda e: (-min(deg[e[0]], deg[e[1]]), rng.random()))

            cliques = greedy_assign(edges, n, order, deadline)
            count = len(cliques)

            if count < best_count:
                best_count = count
                best_cliques = cliques

            if best_count <= m * 0.3:  # good enough, stop early
                break

        if best_cliques is None:
            best_cliques = [list(e) for e in edges]

        return best_cliques

    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed

    deadline1 = start_time + remaining * 0.35
    deadline2 = DEADLINE

    result1 = solve_instance(32, 0.6, 601, deadline1)
    result2 = solve_instance(64, 0.55, 602, deadline2)

    return [result1, result2]
