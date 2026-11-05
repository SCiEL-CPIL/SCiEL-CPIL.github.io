# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return

            if not P:
                return

            # pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def greedy_packing_lower_bound(uncovered_cliques, n, clique_vertices):
        """
        Greedy set packing: repeatedly pick a clique from uncovered that is 'most exclusive'
        (i.e., has fewest vertices that also cover other uncovered cliques), mark all cliques
        sharing any of its vertices as covered by this packing item.
        Returns number of packing items = lower bound on hitting set size.
        """
        if not uncovered_cliques:
            return 0

        # For each vertex, count how many uncovered cliques it belongs to
        vertex_count = [0] * n
        for ci in uncovered_cliques:
            for v in clique_vertices[ci]:
                vertex_count[v] += 1

        # For each clique, compute sum of vertex_counts of its vertices (higher = less exclusive)
        # We want cliques with LOW total vertex_count (more exclusive)
        def clique_exclusivity(ci):
            return sum(vertex_count[v] for v in clique_vertices[ci])

        remaining = set(uncovered_cliques)
        packing = 0

        while remaining:
            # Pick clique with minimum exclusivity (most exclusive)
            best_ci = min(remaining, key=clique_exclusivity)
            packing += 1

            # Remove all cliques that share any vertex with best_ci
            best_vertices = set(clique_vertices[best_ci])
            to_remove = set()
            for ci in remaining:
                if best_vertices & set(clique_vertices[ci]):
                    to_remove.add(ci)
            remaining -= to_remove

        return packing

    def solve_min_hitting_set_bnb(n, cliques, time_budget):
        """
        Exact branch-and-bound for minimum hitting set over clique hypergraph.
        """
        if not cliques:
            return set()

        m = len(cliques)
        clique_vertices = [set(c) for c in cliques]

        # Incidence: for each vertex, list of clique indices it covers
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # Greedy initialization
        def greedy_init():
            uncovered = set(range(m))
            selected = set()
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    gain = sum(1 for i in vertex_cliques[v] if i in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for i in vertex_cliques[best_v]:
                    uncovered.discard(i)
            return selected

        best_solution = greedy_init()
        best_size = len(best_solution)

        # Remove redundant vertices from greedy solution
        changed = True
        while changed:
            changed = False
            for v in list(best_solution):
                test = set(best_solution)
                test.discard(v)
                valid = True
                for i, c in enumerate(cliques):
                    if not any(u in test for u in c):
                        valid = False
                        break
                if valid:
                    best_solution = test
                    best_size = len(best_solution)
                    changed = True
                    break

        # Branch and bound
        # State: (selected_set, uncovered_clique_set)
        # We branch on an uncovered clique: try each vertex in it
        # Use recursion with time checks

        # Precompute for speed: for each clique, sorted list of vertices by coverage (descending)
        clique_vertex_order = []
        for i, c in enumerate(cliques):
            verts = list(c)
            verts.sort(key=lambda v: len(vertex_cliques[v]), reverse=True)
            clique_vertex_order.append(verts)

        # Global best
        global_best = best_size
        global_best_sol = set(best_solution)
        nodes_explored = [0]
        time_exceeded = [False]

        def check_time():
            if time.time() > time_budget:
                time_exceeded[0] = True
                return True
            return False

        def bnb(selected, uncovered):
            if time_exceeded[0]:
                return

            if check_time():
                return

            if not uncovered:
                if len(selected) < global_best:
                    global_best = len(selected)
                    global_best_sol = set(selected)
                return

            # Lower bound
            lb = len(selected) + greedy_packing_lower_bound(uncovered, n, clique_vertices)
            if lb >= global_best:
                return

            nodes_explored[0] += 1
            if nodes_explored[0] % 1000 == 0 and check_time():
                return

            # Pick an uncovered clique to branch on
            # Heuristic: pick the one with fewest vertices (most constrained)
            best_ci = min(uncovered, key=lambda i: len(clique_vertices[i]))
            branch_clique = clique_vertices[best_ci]
            branch_order = clique_vertex_order[best_ci]

            for v in branch_order:
                if time_exceeded[0]:
                    return

                new_selected = selected | {v}
                new_uncovered = set(uncovered)
                # Remove all cliques that v covers
                for ci in vertex_cliques[v]:
                    new_uncovered.discard(ci)

                bnb(new_selected, new_uncovered)

                if time_exceeded[0]:
                    return

        # Start BnB from empty set (or from a reduced set)
        # For efficiency, start from empty and let the bound guide
        initial_uncovered = set(range(m))
        bnb(set(), initial_uncovered)

        if global_best < best_size:
            best_solution = global_best_sol
            best_size = global_best

        # Final cleanup
        changed = True
        while changed:
            changed = False
            for v in list(best_solution):
                test = set(best_solution)
                test.discard(v)
                valid = True
                for i, c in enumerate(cliques):
                    if not any(u in test for u in c):
                        valid = False
                        break
                if valid:
                    best_solution = test
                    changed = True
                    break

        return best_solution

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 30:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 15 - elapsed) / max(1, remaining_instances)
        per_instance = max(2.0, min(per_instance, deadline - 15 - elapsed))

        sol = solve_min_hitting_set_bnb(n, cliques, start_time + elapsed + per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
