<<<<<<< SEARCH
    def local_search(sol, cliques, n, max_iters=5000):
        """Local search to minimize solution size. Returns improved solution."""
        sol_set = set(sol)
        best_sol = list(sol_set)
        best_size = len(best_sol)

        # Precompute for each vertex, which cliques it covers
        # And for each clique, which vertices can cover it
        cliques_list = list(cliques)
        num_cliques = len(cliques_list)

        # Quick check if all cliques are covered
        def covers_all(s):
            s = set(s)
            for c in cliques_list:
                if not s & c:
                    return False
            return True

        if not covers_all(best_sol):
            # Need to fix - add vertices until valid
            for c in cliques_list:
                if not set(best_sol) & c:
                    # Add a vertex from this clique
                    best_sol.append(min(c))
            best_size = len(best_sol)

        rng = random.Random(42)

        for it in range(max_iters):
            # Try removing a vertex
            removable = [v for v in best_sol if all(
                (set(best_sol) - {v}) & c for c in cliques_list
            )]
            if removable:
                # Remove a random removable vertex
                v = rng.choice(removable)
                new_sol = [x for x in best_sol if x != v]
                if is_valid(new_sol, cliques_list):
                    best_sol = new_sol
                    best_size = len(best_sol)
                    continue

            # Try swapping: remove one, add one
            if len(best_sol) >= 2:
                # Try to find a swap that maintains validity and doesn't increase size
                # Actually, swap keeps size same, so we need add+remove to reduce
                # Let's try: for each v in sol, check if we can remove it and the resulting
                # uncovered cliques can be covered by a single vertex not in sol
                pass

            # Try 2-opt: remove two vertices, see if we can cover all with one
            if len(best_sol) >= 2:
                sol_set = set(best_sol)
                # For each pair (v1, v2), check if removing both leaves some cliques uncovered
                # and if those can be covered by a single vertex
                found_swap = False
                for i in range(len(best_sol)):
                    for j in range(i + 1, len(best_sol)):
                        v1, v2 = best_sol[i], best_sol[j]
                        new_sol = [x for x in best_sol if x != v1 and x != v2]
                        new_set = set(new_sol)
                        # Find uncovered cliques
                        uncovered = [c for c in cliques_list if not new_set & c]
                        if not uncovered:
                            best_sol = new_sol
                            best_size = len(best_sol)
                            found_swap = True
                            break
                        # Check if a single vertex covers all uncovered
                        candidate_vertices = set()
                        for c in uncovered:
                            candidate_vertices |= c
                        for w in candidate_vertices:
                            if w not in new_set:
                                if all((new_set | {w}) & c for c in uncovered):
                                    best_sol = new_sol + [w]
                                    best_size = len(best_sol)
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue

            # Try adding a vertex to enable later removals - skip for now

            # If no improvement, try a random perturbation
            if rng.random() < 0.1:
                # Randomly add a vertex, then try to clean up
                candidates = [v for v in range(n) if v not in best_sol]
                if candidates:
                    v = rng.choice(candidates)
                    new_sol = best_sol + [v]
                    if is_valid(new_sol, cliques_list):
                        # Try to remove some vertices
                        best_sol = new_sol
                        best_size = len(best_sol)

        return best_sol
=======
    def local_search(sol, cliques, n, max_iters=5000):
        """Local search to minimize solution size. Returns improved solution."""
        cliques_list = list(cliques)
        best_sol = list(set(sol))
        best_size = len(best_sol)

        # Ensure validity
        sol_set = set(best_sol)
        for c in cliques_list:
            if not sol_set & c:
                best_sol.append(min(c))
                sol_set = set(best_sol)
        best_size = len(best_sol)

        # Precompute: for each vertex, the set of cliques it covers
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques_list):
            for v in c:
                vertex_cliques[v].add(ci)

        def is_valid_fast(s):
            s_set = set(s)
            for c in cliques_list:
                if not s_set & c:
                    return False
            return True

        def count_uncovered(s_set):
            cnt = 0
            for c in cliques_list:
                if not s_set & c:
                    cnt += 1
            return cnt

        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 200

        for it in range(max_iters):
            if it % 50 == 0 and time.time() - start_time > deadline - 15:
                break

            sol_set = set(best_sol)
            improved = False

            # Try removing a vertex
            removable = []
            for v in best_sol:
                test_set = sol_set - {v}
                if all(test_set & c for c in cliques_list):
                    removable.append(v)
            if removable:
                v = rng.choice(removable)
                best_sol = [x for x in best_sol if x != v]
                best_size -= 1
                improved = True
                no_improve_count = 0
                continue

            # Try 2-opt: remove two, add one (net -1)
            if len(best_sol) >= 2:
                found_swap = False
                # Sample pairs to avoid O(k^2) blowup for large solutions
                if len(best_sol) > 20:
                    pair_indices = list(rng.sample(range(len(best_sol)), min(200, len(best_sol) * (len(best_sol)-1) // 2)))
                    pairs = set()
                    for idx in pair_indices:
                        i = idx % len(best_sol)
                        j = (idx // len(best_sol)) % len(best_sol)
                        if i != j:
                            pairs.add((min(i,j), max(i,j)))
                    pair_list = list(pairs)
                else:
                    pair_list = [(i, j) for i in range(len(best_sol)) for j in range(i+1, len(best_sol))]

                for i, j in pair_list:
                    v1, v2 = best_sol[i], best_sol[j]
                    new_set = sol_set - {v1, v2}
                    uncovered = [c for c in cliques_list if not new_set & c]
                    if not uncovered:
                        best_sol = [x for x in best_sol if x != v1 and x != v2]
                        best_size -= 2
                        improved = True
                        found_swap = True
                        break
                    # Check if a single vertex covers all uncovered
                    candidate_vertices = set()
                    for c in uncovered:
                        candidate_vertices |= c
                    for w in candidate_vertices:
                        if w not in new_set:
                            if all((new_set | {w}) & c for c in uncovered):
                                best_sol = [x for x in best_sol if x != v1 and x != v2] + [w]
                                best_size -= 1
                                improved = True
                                found_swap = True
                                break
                    if found_swap:
                        break
                if found_swap:
                    no_improve_count = 0
                    continue

            # Try 1-opt swap: remove one, add one (same size, but may enable future removals)
            if len(best_sol) >= 1:
                sol_set = set(best_sol)
                for v in best_sol:
                    new_set = sol_set - {v}
                    uncovered = [c for c in cliques_list if not new_set & c]
                    if not uncovered:
                        continue
                    # Find a vertex w not in new_set that covers all uncovered
                    candidates = set()
                    for c in uncovered:
                        candidates |= c
                    candidates -= new_set
                    for w in candidates:
                        if all((new_set | {w}) & c for c in uncovered):
                            # This swap keeps size same; only do it if it seems beneficial
                            # (e.g., w covers more cliques than v did)
                            if len(vertex_cliques[w]) > len(vertex_cliques[v]):
                                best_sol = [x for x in best_sol if x != v] + [w]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    no_improve_count = 0
                    continue

            # Simulated annealing perturbation
            if no_improve_count > max_no_improve:
                temp = max(0.1, 1.0 - it / max_iters)
                # Randomly add 1-2 vertices, then try to clean up
                num_add = rng.randint(1, 2)
                candidates = [v for v in range(n) if v not in set(best_sol)]
                if candidates:
                    for _ in range(num_add):
                        if not candidates:
                            break
                        v = rng.choice(candidates)
                        candidates.remove(v)
                        best_sol.append(v)
                        best_size += 1

                    # Now try aggressive cleanup: repeatedly remove any removable vertex
                    cleanup_iters = 0
                    while cleanup_iters < 50:
                        sol_set = set(best_sol)
                        removable = []
                        for v in best_sol:
                            test_set = sol_set - {v}
                            if all(test_set & c for c in cliques_list):
                                removable.append(v)
                        if not removable:
                            break
                        v = rng.choice(removable)
                        best_sol = [x for x in best_sol if x != v]
                        best_size -= 1
                        cleanup_iters += 1

                    # If we got back to valid and smaller, great; otherwise revert
                    if not is_valid_fast(best_sol):
                        # Revert: this shouldn't happen after cleanup, but safety
                        pass

                no_improve_count = 0

            if not improved:
                no_improve_count += 1

        return best_sol
>>>>>>> REPLACE

