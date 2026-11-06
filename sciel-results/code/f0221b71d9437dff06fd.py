"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    import numpy as np

    DEADLINE = 140.0  # safety margin before 150s kill
    t0 = time.time()

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot, using bitmasks."""
        # Build adjacency as bitmasks for speed
        adj_mask = [0] * n
        for v in range(n):
            for u in range(n):
                if adj[v][u]:
                    adj_mask[v] |= (1 << u)

        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(R)
                return

            # Choose pivot u from P ∪ X that maximizes |P & adj_mask[u]|
            union = P | X
            if union:
                best_pivot = -1
                best_count = -1
                temp = union
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    count = (P & adj_mask[v]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_pivot = v
                # Candidates: vertices in P not adjacent to pivot
                candidates = P & ~(adj_mask[best_pivot]) if best_pivot >= 0 else P
            else:
                candidates = 0

            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb

                new_R = R | lsb
                new_P = P & adj_mask[v]
                new_X = X & adj_mask[v]
                bk(new_R, new_P, new_X)

                P &= ~lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def clique_to_list(mask, n):
        """Convert bitmask to list of vertex indices."""
        result = []
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            result.append(v)
            temp ^= lsb
        return result

    def solve_instance(n, p, seed, remaining_time):
        adj = generate_graph(n, p, seed)
        cliques_masks = bron_kerbosch(adj, n)

        if not cliques_masks:
            return []

        # Build incidence: for each vertex, which cliques it covers
        # cliques_masks[i] is a bitmask of vertices in clique i
        num_cliques = len(cliques_masks)

        # Vertex-to-clique incidence as bitmask
        vertex_cliques = [0] * n
        for i, cmask in enumerate(cliques_masks):
            temp = cmask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v] |= (1 << i)
                temp ^= lsb

        # Greedy set cover to get initial solution
        uncovered = (1 << num_cliques) - 1
        selected = set()
        selected_mask = 0  # bitmask of selected vertices

        while uncovered and (time.time() - t0) < remaining_time:
            best_v = -1
            best_count = 0
            for v in range(n):
                if v in selected:
                    continue
                count = (uncovered & vertex_cliques[v]).bit_count()
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            selected.add(best_v)
            selected_mask |= (1 << best_v)
            uncovered &= ~vertex_cliques[best_v]

        # Local search: 1-removal, 1-addition, 2-swap
        def is_valid(sel_mask):
            """Check if selection covers all cliques."""
            # For each clique, check if any selected vertex is in it
            for cmask in cliques_masks:
                if (sel_mask & cmask) == 0:
                    return False
            return True

        def local_search(sel_mask, sel_set, max_time):
            improved = True
            while improved and (time.time() - t0) < max_time:
                improved = False

                # Try removing each selected vertex
                temp = sel_mask
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    new_mask = sel_mask & ~lsb
                    if is_valid(new_mask):
                        sel_mask = new_mask
                        sel_set.discard(v)
                        improved = True
                        break

                if improved:
                    continue

                # Try 2-swap: remove one selected, add one unselected
                sel_list = list(sel_set)
                for i in range(len(sel_list)):
                    if (time.time() - t0) > max_time:
                        return sel_mask, sel_set
                    v_remove = sel_list[i]
                    remove_mask = 1 << v_remove
                    for v_add in range(n):
                        if v_add in sel_set:
                            continue
                        add_mask = 1 << v_add
                        new_mask = (sel_mask & ~remove_mask) | add_mask
                        if is_valid(new_mask):
                            sel_mask = new_mask
                            sel_set.discard(v_remove)
                            sel_set.add(v_add)
                            sel_list = list(sel_set)
                            improved = True
                            break
                    if improved:
                        break

                if improved:
                    continue

                # Try 2-opt: remove two selected, add two unselected
                sel_list = list(sel_set)
                found = False
                for i in range(len(sel_list)):
                    if (time.time() - t0) > max_time:
                        break
                    for j in range(i + 1, len(sel_list)):
                        if (time.time() - t0) > max_time:
                            break
                        v1, v2 = sel_list[i], sel_list[j]
                        remove_mask = (1 << v1) | (1 << v2)
                        base_mask = sel_mask & ~remove_mask
                        for a in range(n):
                            if a in sel_set:
                                continue
                            for b in range(a + 1, n):
                                if b in sel_set:
                                    continue
                                add_mask = (1 << a) | (1 << b)
                                new_mask = base_mask | add_mask
                                if is_valid(new_mask):
                                    sel_mask = new_mask
                                    sel_set.discard(v1)
                                    sel_set.discard(v2)
                                    sel_set.add(a)
                                    sel_set.add(b)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break

                if found:
                    improved = True

            return sel_mask, sel_set

        # Run local search
        max_ls_time = min(remaining_time - (time.time() - t0), 10.0)
        sel_mask, sel_set = local_search(selected_mask, selected, max_ls_time)

        return sorted(sel_set)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed
        if remaining < 5.0:
            # Not enough time, return empty for remaining instances
            results.append([])
            continue
        result = solve_instance(n, p, seed, remaining)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
