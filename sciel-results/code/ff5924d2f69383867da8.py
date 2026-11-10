import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return

        P_union_X = P | X
        if P_union_X:
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            candidates = P - adj[pivot] if pivot is not None else P
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges, time_budget):
        if not edges:
            return []

        adj = build_adj(n, edges)
        maximal_cliques = []
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)

        if not maximal_cliques:
            return []

        n_cliques = len(maximal_cliques)
        # For each vertex, list of clique indices it hits
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # Precompute clique vertex lists for fast access
        clique_lists = [list(c) for c in maximal_cliques]

        def is_valid_transversal(candidate_set):
            for c in maximal_cliques:
                if not (c & candidate_set):
                    return False
            return True

        def greedy_init(rng):
            unhit_mask = (1 << n_cliques) - 1
            transversal = set()
            while unhit_mask:
                if time.time() > time_budget:
                    break
                freq = [0] * n
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    for v in clique_lists[idx]:
                        freq[v] += 1
                if not any(freq):
                    break
                # Find top-2 frequencies for regret-based scoring
                best_v = None
                best_score = None
                # Collect all candidates with their scores
                cands = []
                for v in range(n):
                    f = freq[v]
                    if f == 0:
                        continue
                    # Add small random perturbation for diversification
                    noise = rng.random() * 0.1
                    score = f + noise
                    cands.append((score, v))
                cands.sort(reverse=True)
                if not cands:
                    break
                best_v = cands[0][1]
                transversal.add(best_v)
                for ci in vertex_cliques[best_v]:
                    unhit_mask &= ~(1 << ci)
            # Ensure validity
            if unhit_mask:
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    transversal.add(clique_lists[idx][0])
            return transversal

        def local_search(transversal, budget):
            improved = True
            max_passes = 100
            pass_count = 0
            while improved and time.time() < budget and pass_count < max_passes:
                improved = False
                pass_count += 1

                # 1. Greedy Removal
                for v in list(transversal):
                    if time.time() > budget:
                        return transversal
                    candidate = transversal - {v}
                    if is_valid_transversal(candidate):
                        transversal = candidate
                        improved = True

                if not transversal:
                    break

                # 2. k-for-1 Replacement (k=2, k=3)
                trans_list = list(transversal)
                size = len(trans_list)

                # k=2
                for i in range(size):
                    if time.time() > budget:
                        return transversal
                    for j in range(i + 1, size):
                        v1 = trans_list[i]
                        v2 = trans_list[j]
                        base_candidate = transversal - {v1, v2}
                        unhit_by_base = []
                        for c in maximal_cliques:
                            if not (c & base_candidate):
                                unhit_by_base.append(c)
                        if not unhit_by_base:
                            transversal = base_candidate
                            improved = True
                            break
                        potential_w = set(unhit_by_base[0])
                        for c in unhit_by_base[1:]:
                            potential_w &= c
                        potential_w.discard(v1)
                        potential_w.discard(v2)
                        if potential_w:
                            w = next(iter(potential_w))
                            transversal = base_candidate | {w}
                            improved = True
                            break
                    if improved:
                        break

                if improved:
                    continue

                # k=3
                if len(transversal) >= 3:
                    trans_list = list(transversal)
                    size = len(trans_list)
                    for i in range(size):
                        if time.time() > budget:
                            return transversal
                        for j in range(i + 1, size):
                            for k in range(j + 1, size):
                                v1, v2, v3 = trans_list[i], trans_list[j], trans_list[k]
                                base_candidate = transversal - {v1, v2, v3}
                                unhit_by_base = []
                                for c in maximal_cliques:
                                    if not (c & base_candidate):
                                        unhit_by_base.append(c)
                                if not unhit_by_base:
                                    transversal = base_candidate
                                    improved = True
                                    break
                                potential_w = set(unhit_by_base[0])
                                for c in unhit_by_base[1:]:
                                    potential_w &= c
                                potential_w.discard(v1)
                                potential_w.discard(v2)
                                potential_w.discard(v3)
                                if potential_w:
                                    w = next(iter(potential_w))
                                    transversal = base_candidate | {w}
                                    improved = True
                                    break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break

            return transversal

        def branch_and_bound(transversal, budget):
            best_size = len(transversal)
            best_solution = set(transversal)

            # Sort cliques by size descending
            sorted_cliques = sorted(enumerate(maximal_cliques), key=lambda x: -len(x[1]))

            # Precompute for each vertex: bitmask of cliques it hits (in sorted order)
            vertex_clique_mask = [0] * n
            for new_idx, (old_idx, c) in enumerate(sorted_cliques):
                for v in c:
                    vertex_clique_mask[v] |= (1 << new_idx)

            # Initial unhit mask
            initial_unhit = 0
            for i, c in enumerate(sorted_cliques):
                if not (c & transversal):
                    initial_unhit |= (1 << i)

            # If initial is already valid, we're good
            if initial_unhit == 0:
                return best_solution

            def search(unhit_mask, current_set, current_size, depth):
                nonlocal best_size, best_solution

                if time.time() > budget:
                    return

                if current_size >= best_size:
                    return

                if not unhit_mask:
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = set(current_set)
                    return

                # Lower bound: at least 1 more vertex needed
                if current_size + 1 >= best_size:
                    return

                # Choose clique with fewest candidate vertices
                unhit_list = []
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    unhit_list.append(idx)

                if not unhit_list:
                    return

                best_clique_idx = None
                best_candidates = None
                min_cand_count = float('inf')

                for ci in unhit_list:
                    c = sorted_cliques[ci][1]
                    candidates = [v for v in c if v not in current_set]
                    if len(candidates) < min_cand_count:
                        min_cand_count = len(candidates)
                        best_clique_idx = ci
                        best_candidates = candidates
                        if min_cand_count <= 1:
                            break

                if not best_candidates:
                    return

                # Sort candidates by frequency in unhit cliques (descending)
                def candidate_freq(v):
                    count = 0
                    m = unhit_mask
                    while m:
                        lowbit = m & (-m)
                        idx = lowbit.bit_length() - 1
                        m ^= lowbit
                        if v in sorted_cliques[idx][1]:
                            count += 1
                    return -count

                best_candidates.sort(key=candidate_freq)

                for v in best_candidates:
                    if time.time() > budget:
                        return
                    new_unhit = unhit_mask & ~vertex_clique_mask[v]
                    new_set = current_set | {v}
                    search(new_unhit, new_set, current_size + 1, depth + 1)

            search(initial_unhit, set(transversal), len(transversal), 0)
            return best_solution

        # Multi-start local search
        best_sol = None
        best_size = float('inf')

        # Allocate time: 60% for multi-start, 40% for B&B polish
        multi_start_budget = time_budget * 0.6
        bb_budget = time_budget * 0.4

        # Number of restarts based on time available
        num_restarts = max(3, int((multi_start_budget - (time.time() - start_time)) / 2.0))
        num_restarts = min(num_restarts, 30)

        for r in range(num_restarts):
            if time.time() > multi_start_budget:
                break
            rng = Random(12345 + r * 7919)
            sol = greedy_init(rng)
            sol = local_search(sol, multi_start_budget)
            if len(sol) < best_size:
                best_size = len(sol)
                best_sol = sol

        # Final B&B polish on best solution
        if best_sol is not None and time.time() < deadline - 5:
            polished = branch_and_bound(best_sol, deadline - 5)
            if len(polished) < best_size:
                best_sol = polished

        return sorted(best_sol) if best_sol else []

    # Allocate time to instances based on difficulty
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    # Difficulty heuristic: n * p (approx expected edges)
    difficulties = [n * p for n, p, s in instances]
    total_diff = sum(difficulties)

    results = []
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed

    for i, (n, p, seed) in enumerate(instances):
        # Allocate proportional time, but ensure minimum
        alloc = remaining * (difficulties[i] / total_diff)
        alloc = max(alloc, 5)  # at least 5 seconds
        # But don't exceed what's left
        alloc = min(alloc, remaining - 5)  # leave 5s for remaining instances

        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges, alloc)
        results.append(sol)

        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed

    return results
