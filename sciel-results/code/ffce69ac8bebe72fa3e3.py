"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def build_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []

        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Choose pivot from P ∪ X to minimize branching
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                R2 = R | {v}
                P2 = P & adj[v]
                X2 = X & adj[v]
                expand(R2, P2, X2)
                P -= {v}
                X |= {v}

        expand(set(), set(range(n)), set())
        return maximal_cliques

    def greedy_hitting_set(cliques, adj, n):
        """
        Greedy set cover: each vertex covers all cliques it belongs to.
        Returns a hitting set (list of vertices).
        """
        # Map each clique to its index
        cliques = list(cliques)
        num_cliques = len(cliques)
        if num_cliques == 0:
            return []

        # For each vertex, find which cliques it hits
        vertex_to_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)

        uncovered = set(range(num_cliques))
        selected = []

        while uncovered:
            if time.time() > deadline:
                break
            # Find vertex that covers most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.append(best_v)
            uncovered -= vertex_to_cliques[best_v]

        return selected

    def local_search_improve(hitting_set, cliques, adj, n, time_budget=5.0):
        """
        Try to remove vertices from the hitting set if possible,
        and swap vertices to reduce size.
        """
        hitting_set = list(hitting_set)
        end_time = time.time() + time_budget
        improved = True

        while improved and time.time() < end_time:
            improved = False

            # Try removing each vertex
            for i in range(len(hitting_set)):
                candidate = hitting_set[:i] + hitting_set[i+1:]
                # Check if candidate still hits all cliques
                candidate_set = set(candidate)
                valid = True
                for clique in cliques:
                    if not (set(clique) & candidate_set):
                        valid = False
                        break
                if valid:
                    hitting_set = candidate
                    improved = True
                    break

            # Try swapping one vertex with another to reduce size
            # (This is more complex; skip for now and rely on removal)

        return hitting_set

    def branch_and_bound_hitting_set(cliques, adj, n, time_limit=10.0):
        """
        Branch and bound to find a minimal hitting set.
        Uses greedy solution as initial upper bound.
        """
        cliques = list(cliques)
        num_cliques = len(cliques)
        if num_cliques == 0:
            return []

        # Initial greedy solution
        best = greedy_hitting_set(cliques, adj, n)
        best_size = len(best)
        if best_size <= 1:
            return best

        # Track uncovered cliques for pruning
        # State: (uncovered_cliques_frozen_set, current_selection_frozen_set)
        # We'll use a simpler approach: recursive with memoization-like pruning

        start_bb = time.time()
        best_solution = list(best)
        best_size = len(best)

        # Sort vertices by frequency in cliques (more frequent first)
        vertex_freq = [0] * n
        for clique in cliques:
            for v in clique:
                vertex_freq[v] += 1
        vertices_sorted = sorted(range(n), key=lambda v: -vertex_freq[v])

        # For each vertex, precompute which cliques it covers
        vertex_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)

        # Convert cliques to frozensets for hashing
        cliques_fs = [frozenset(c) for c in cliques]

        def bb(uncovered, selected, depth):
            nonlocal best_solution, best_size
            if time.time() - start_bb > time_limit:
                return

            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = list(selected)
                return

            # Lower bound: need at least ceil(|uncovered| / max_cover) more vertices
            # where max_cover is max number of uncovered cliques any single vertex can cover
            max_cover = 0
            for v in vertices_sorted:
                cover = len(vertex_cliques[v] & uncovered)
                if cover > max_cover:
                    max_cover = cover
                if max_cover >= len(uncovered):
                    break
            if max_cover == 0:
                return
            import math
            lb = len(selected) + math.ceil(len(uncovered) / max_cover)
            if lb >= best_size:
                return

            # Branch on vertices that cover at least one uncovered clique
            # Pick a clique to branch on
            target_clique = next(iter(uncovered))
            # Vertices in this clique
            target_vertices = [v for v in cliques_fs[target_clique]
                               if len(vertex_cliques[v] & uncovered) > 0]
            if not target_vertices:
                return

            # Sort by coverage
            target_vertices.sort(key=lambda v: -len(vertex_cliques[v] & uncovered))

            for v in target_vertices:
                if time.time() - start_bb > time_limit:
                    return
                new_uncovered = uncovered - vertex_cliques[v]
                new_selected = selected | {v}
                bb(new_uncovered, new_selected, depth + 1)

        bb(set(range(num_cliques)), set(), 0)
        return best_solution

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]

    for n, p, seed in instances:
        if time.time() > deadline:
            # Return what we have so far with empty lists for remaining
            results.append([])
            continue

        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)

        if not cliques:
            results.append([])
            continue

        # Get a good initial solution via greedy
        initial = greedy_hitting_set(cliques, adj, n)

        # Try branch and bound with time limit
        bb_time = min(20.0, max(1.0, (deadline - time.time()) / 3))
        solution = branch_and_bound_hitting_set(cliques, adj, n, time_limit=bb_time)

        # Local search to try to improve
        solution = local_search_improve(solution, cliques, adj, n, time_budget=2.0)

        results.append(sorted(solution))

    return results
# EVOLVE-BLOCK-END
