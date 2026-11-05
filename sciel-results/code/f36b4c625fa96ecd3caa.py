"""Strategy: Greedy set cover initialization + simulated annealing local search on the clique hypergraph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

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

            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return

            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            P_set = set(P)
            X_set = set(X)

            if best_pivot is not None:
                candidates = [v for v in P_set if v not in adj[best_pivot]]
            else:
                candidates = list(P_set)

            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                bk(new_R, new_P, new_X)
                P_set.remove(v)
                X_set.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_hitting_set_sa(n, cliques, time_budget):
        """
        Greedy initialization + simulated annealing to minimize hitting set size.
        State: frozenset of selected vertices.
        Objective: minimize |selected| subject to every clique being hit.
        Moves: add a random uncovered vertex, remove a redundant vertex, swap.
        """
        if not cliques:
            return set()

        m = len(cliques)

        # Precompute: for each vertex, which cliques it covers (as list of indices)
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
                    gain = len([i for i in vertex_cliques[v] if i in uncovered])
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for i in vertex_cliques[best_v]:
                    if i in uncovered:
                        uncovered.discard(i)
            return selected

        # Check if a set is valid (hits all cliques)
        def is_valid(selected):
            for c in cliques:
                if not any(v in selected for v in c):
                    return False
            return True

        # Get uncovered cliques for a selected set
        def get_uncovered(selected):
            uncovered = []
            for i, c in enumerate(cliques):
                if not any(v in selected for v in c):
                    uncovered.append(i)
            return uncovered

        # Simulated annealing
        best_solution = greedy_init()
        best_size = len(best_solution)
        current = set(best_solution)

        # Temperature schedule
        T_max = 1.0
        T_min = 0.01
        alpha = 0.9995  # cooling rate
        T = T_max

        rng = random.Random(42)

        while time.time() < time_budget and T > T_min:
            # Generate a move
            move_type = rng.random()
            new_current = set(current)

            if move_type < 0.3:
                # Add a vertex
                candidates = [v for v in range(n) if v not in current]
                if candidates:
                    # Prefer vertices that cover uncovered cliques
                    unc = get_uncovered(current)
                    if unc:
                        weighted = []
                        for v in candidates:
                            w = len([i for i in vertex_cliques[v] if i in unc])
                            weighted.extend([v] * max(1, w))
                        if weighted:
                            v = rng.choice(weighted)
                        else:
                            v = rng.choice(candidates)
                    else:
                        v = rng.choice(candidates)
                    new_current.add(v)

            elif move_type < 0.6:
                # Remove a vertex
                if current:
                    # Prefer vertices that are "redundant" (their cliques are covered by others)
                    candidates = []
                    for v in current:
                        # Check if v is the only vertex covering any clique
                        is_only = False
                        for i in vertex_cliques[v]:
                            c = cliques[i]
                            if not any(u in current for u in c if u != v):
                                is_only = True
                                break
                        if not is_only:
                            candidates.append(v)
                    if candidates:
                        v = rng.choice(candidates)
                        new_current.discard(v)
                    else:
                        # All vertices are critical; try removing one anyway
                        v = rng.choice(list(current))
                        new_current.discard(v)

            elif move_type < 0.8:
                # Swap: remove one, add one
                if current:
                    # Remove a non-critical vertex if possible
                    rem_candidates = []
                    for v in current:
                        is_only = False
                        for i in vertex_cliques[v]:
                            c = cliques[i]
                            if not any(u in current for u in c if u != v):
                                is_only = True
                                break
                        if not is_only:
                            rem_candidates.append(v)
                    if rem_candidates:
                        v_rem = rng.choice(rem_candidates)
                        new_current.discard(v_rem)
                        # Add a vertex that helps
                        unc = get_uncovered(new_current)
                        add_candidates = [v for v in range(n) if v not in new_current]
                        if add_candidates:
                            if unc:
                                weighted = []
                                for v in add_candidates:
                                    w = len([i for i in vertex_cliques[v] if i in unc])
                                    weighted.extend([v] * max(1, w))
                                if weighted:
                                    v_add = rng.choice(weighted)
                                else:
                                    v_add = rng.choice(add_candidates)
                            else:
                                v_add = rng.choice(add_candidates)
                            new_current.add(v_add)
                        else:
                            # No vertices to add; revert
                            new_current.add(v_rem)
                    else:
                        # No non-critical vertices; do a random swap
                        v_rem = rng.choice(list(current))
                        new_current.discard(v_rem)
                        add_candidates = [v for v in range(n) if v not in new_current]
                        if add_candidates:
                            v_add = rng.choice(add_candidates)
                            new_current.add(v_add)
                        else:
                            new_current.add(v_rem)

            else:
                # 2-opt: remove two, add two
                if len(current) >= 2:
                    # Find two non-critical vertices
                    rem_candidates = []
                    for v in current:
                        is_only = False
                        for i in vertex_cliques[v]:
                            c = cliques[i]
                            if not any(u in current for u in c if u != v):
                                is_only = True
                                break
                        if not is_only:
                            rem_candidates.append(v)
                    if len(rem_candidates) >= 2:
                        v1, v2 = rng.sample(rem_candidates, 2)
                        new_current.discard(v1)
                        new_current.discard(v2)
                        # Add two vertices
                        unc = get_uncovered(new_current)
                        add_candidates = [v for v in range(n) if v not in new_current]
                        if len(add_candidates) >= 2:
                            if unc:
                                weighted = []
                                for v in add_candidates:
                                    w = len([i for i in vertex_cliques[v] if i in unc])
                                    weighted.extend([v] * max(1, w))
                                if weighted:
                                    picks = rng.sample(weighted, min(2, len(weighted)))
                                    # Deduplicate
                                    unique_picks = list(dict.fromkeys(picks))
                                    while len(unique_picks) < 2 and len(add_candidates) > len(unique_picks):
                                        extra = rng.choice(add_candidates)
                                        if extra not in unique_picks:
                                            unique_picks.append(extra)
                                    for v in unique_picks:
                                        new_current.add(v)
                                else:
                                    v1_add, v2_add = rng.sample(add_candidates, 2)
                                    new_current.add(v1_add)
                                    new_current.add(v2_add)
                            else:
                                v1_add, v2_add = rng.sample(add_candidates, 2)
                                new_current.add(v1_add)
                                new_current.add(v2_add)
                        else:
                            # Not enough to add; revert
                            new_current.add(v1)
                            new_current.add(v2)
                    else:
                        pass  # No valid 2-opt move

            # Evaluate
            new_size = len(new_current)
            old_size = len(current)

            # Check validity
            if is_valid(new_current):
                # Accept if better, or accept worse with probability exp(-delta/T)
                delta = new_size - old_size
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = new_current
                    if new_size < best_size:
                        best_size = new_size
                        best_solution = set(current)
            else:
                # Invalid: only accept if we can quickly repair, or reject
                # For simplicity, reject invalid states
                pass

            T *= alpha

        # Final local search: try to remove any redundant vertices
        for v in list(best_solution):
            test = set(best_solution)
            test.discard(v)
            if is_valid(test):
                best_solution = test

        # Try to add vertices to reduce... no, we want to minimize size.
        # Do a final pass: for each vertex in best_solution, try removing it
        changed = True
        while changed:
            changed = False
            for v in list(best_solution):
                test = set(best_solution)
                test.discard(v)
                if is_valid(test):
                    best_solution = test
                    changed = True

        return best_solution

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 20:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        # Allocate remaining time
        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 10 - elapsed) / max(1, remaining_instances)
        per_instance = max(1.0, min(per_instance, deadline - 10 - elapsed))

        sol = solve_hitting_set_sa(n, cliques, start_time + elapsed + per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
