import itertools
import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as lists of triple indices.
    transversals = []
    seen = set()
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        remaining = [1, 2, 3, 4, 5]
        for j in remaining:
            rest = [x for x in remaining if x != j]
            for k in range(3):
                if k == 0:
                    p2, p3 = (rest[0], rest[1]), (rest[2], rest[3])
                elif k == 1:
                    p2, p3 = (rest[0], rest[2]), (rest[1], rest[3])
                else:
                    p2, p3 = (rest[0], rest[3]), (rest[1], rest[2])
                matching = tuple(sorted(((sp[0], sp[j]), (sp[p2[0]], sp[p2[1]]), (sp[p3[0]], sp[p3[1]]))))
                matchings.add(matching)

        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            triple_indices = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                triple_indices.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(triple_indices))
            if key not in seen:
                seen.add(key)
                transversals.append(triple_indices)
    num_trans = len(transversals)

    # For each triple, the list of transversals it participates in.
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tlist in enumerate(transversals):
        for idx in tlist:
            triple_trans[idx].append(ti)

    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_trans
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        for idx in candidates:
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
        return sol, counts, sum(sol)

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
                added += 1
            else:
                new_candidates.append(idx)
        return added, new_candidates

    def add_triple(idx, sol, counts):
        sol[idx] = True
        for ci in triple_trans[idx]:
            counts[ci] += 1

    def remove_triple(idx, sol, counts):
        sol[idx] = False
        for ci in triple_trans[idx]:
            counts[ci] -= 1

    def is_safe_to_add(idx, counts, sol):
        if sol[idx]:
            return True
        for ci in triple_trans[idx]:
            if counts[ci] == 7:
                return False
        return True

    def polish(sol, counts, current_count, max_swap_iters=40):
        """High-precision polish: add safe triples, then 1-out-1-in and 2-out-1-in swaps."""
        # Phase 1: Add all safe triples
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        added, _ = try_add_candidates(sol, counts, candidates)
        current_count += added

        if time.time() > deadline - 2.0:
            return current_count

        improved = True
        for _ in range(max_swap_iters):
            if time.time() > deadline - 1.5:
                break
            if not improved:
                break
            improved = False

            present = [i for i in range(num_triples) if sol[i]]
            absent = [i for i in range(num_triples) if not sol[i]]
            if not present or not absent:
                break
            random.shuffle(present)
            random.shuffle(absent)

            # Try 1-out-1-in swaps
            found_swap = False
            for rem in present:
                if time.time() > deadline - 1.5:
                    break
                freed_trans = set()
                for ci in triple_trans[rem]:
                    if counts[ci] == 7:
                        freed_trans.add(ci)
                if not freed_trans:
                    continue

                for add in absent:
                    safe = True
                    for ci in triple_trans[add]:
                        if counts[ci] == 7 and ci not in freed_trans:
                            safe = False
                            break
                    if safe:
                        remove_triple(rem, sol, counts)
                        add_triple(add, sol, counts)
                        # Try to add more after swap
                        remaining_absent = [i for i in range(num_triples) if not sol[i]]
                        random.shuffle(remaining_absent)
                        added2, _ = try_add_candidates(sol, counts, remaining_absent)
                        current_count += added2
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break

            # Try 2-out-1-in swaps if 1-out-1-in didn't help
            if not improved and time.time() < deadline - 1.0:
                present = [i for i in range(num_triples) if sol[i]]
                absent = [i for i in range(num_triples) if not sol[i]]
                if len(present) >= 2 and absent:
                    random.shuffle(present)
                    random.shuffle(absent)
                    for i in range(min(len(present), 60)):
                        for j in range(i + 1, min(len(present), 60)):
                            if time.time() > deadline - 1.0:
                                break
                            rem1, rem2 = present[i], present[j]
                            freed_trans = set()
                            for ci in triple_trans[rem1]:
                                if counts[ci] == 7:
                                    freed_trans.add(ci)
                            for ci in triple_trans[rem2]:
                                if counts[ci] == 7:
                                    freed_trans.add(ci)
                            if not freed_trans:
                                continue

                            for add in absent[:40]:
                                safe = True
                                for ci in triple_trans[add]:
                                    if counts[ci] == 7 and ci not in freed_trans:
                                        safe = False
                                        break
                                if safe:
                                    remove_triple(rem1, sol, counts)
                                    remove_triple(rem2, sol, counts)
                                    add_triple(add, sol, counts)
                                    current_count -= 1

                                    remaining_absent = [i for i in range(num_triples) if not sol[i]]
                                    random.shuffle(remaining_absent)
                                    added2, _ = try_add_candidates(sol, counts, remaining_absent)
                                    current_count += added2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

        return current_count

    best_solution = None
    best_count = -1
    best_counts = None

    # Main search: fast restarts with local search, track best
    num_restarts = 120
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        sol, counts, current_count = greedy_init()

        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)

        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            best_counts = counts[:]

        stagnation = 0
        max_local_iters = 80
        for _ in range(max_local_iters):
            if time.time() > deadline - 5.0:
                break

            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added

            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                    best_counts = counts[:]
                continue

            # Find tight configurations (count == 7)
            tight_triples = []
            for ci, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in transversals[ci]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)

            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                active = [i for i in range(num_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)

            remove_triple(kick_idx, sol, counts)
            current_count -= 1
            candidates.append(kick_idx)

            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added

            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                    best_counts = counts[:]
            else:
                stagnation += 1
                if stagnation > 40:
                    break

    if best_solution is None:
        best_solution, best_counts, best_count = greedy_init()

    # Final high-precision polish on the best solution
    if time.time() < deadline - 3.0:
        final_count = polish(best_solution, best_counts, best_count, max_swap_iters=50)
        if final_count > best_count:
            best_count = final_count

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution[:]
    final_counts = best_counts[:]
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            for t_idx in transversals[ci]:
                if final_sol[t_idx]:
                    final_sol[t_idx] = False
                    for ci2 in triple_trans[t_idx]:
                        final_counts[ci2] -= 1
                    break

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
