import itertools
import random
import time
import heapq
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n  # 729

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    neighbors_list = [None] * total
    neighbors_set = [None] * total
    neighbors_arr = [None] * total

    for idx in range(total):
        w = idx_to_tuple(idx)
        nbs = [idx]
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                nbs.append(tuple_to_idx(tuple(new_w)))
        neighbors_list[idx] = nbs
        neighbors_set[idx] = set(nbs)
        neighbors_arr[idx] = np.array(nbs, dtype=np.int32)

    # Tetracode structure: 9x9 grid
    cell_options = {}
    for a in range(3):
        for b in range(3):
            opts = []
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a + b) % 3, (a + 2 * b) % 3, x, y)
                    cw_idx = tuple_to_idx(cw)
                    covered = neighbors_set[cw_idx]
                    covered_arr = neighbors_arr[cw_idx]
                    opts.append((cw_idx, covered, covered_arr))
            cell_options[(a, b)] = opts

    def tetracode_full():
        result = []
        for a in range(3):
            for b in range(3):
                for x in range(3):
                    for y in range(3):
                        cw = (a, b, (a + b) % 3, (a + 2 * b) % 3, x, y)
                        result.append(tuple_to_idx(cw))
        return result

    def tetracode_partial(k_cells):
        all_cells = [(a, b) for a in range(3) for b in range(3)]
        random.shuffle(all_cells)
        chosen = all_cells[:k_cells]
        result = []
        for (a, b) in chosen:
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a + b) % 3, (a + 2 * b) % 3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    def tetracode_random_options():
        result = []
        for a in range(3):
            for b in range(3):
                num_opts = random.randint(1, 9)
                opts = random.sample(range(9), num_opts)
                for opt_idx in opts:
                    x, y = divmod(opt_idx, 3)
                    cw = (a, b, (a + b) % 3, (a + 2 * b) % 3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    def greedy_cover():
        uncovered = np.ones(total, dtype=bool)
        code = []
        while uncovered.any():
            best_cand = -1
            best_count = -1
            candidates = list(range(total))
            random.shuffle(candidates)
            for cand in candidates:
                if not uncovered.any():
                    break
                covered_here = int(np.sum(uncovered[neighbors_arr[cand]]))
                if covered_here > best_count:
                    best_count = covered_here
                    best_cand = cand
            if best_cand < 0:
                break
            code.append(best_cand)
            uncovered[neighbors_arr[best_cand]] = False
        return code

    def ensure_valid(code):
        """Add codewords until coverage is complete."""
        code = list(dict.fromkeys(code))
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        uncovered = np.where(cover_count == 0)[0]
        while len(uncovered) > 0:
            best_cand = -1
            best_count = -1
            for cand in uncovered:
                covered_here = len(neighbors_set[cand] & set(uncovered.tolist()))
                if covered_here > best_count:
                    best_count = covered_here
                    best_cand = cand
            if best_cand < 0:
                break
            code.append(best_cand)
            cover_count[neighbors_arr[best_cand]] += 1
            uncovered = np.where(cover_count == 0)[0]
        return code

    def fast_local_search(code, time_limit):
        """Local search with priority-queue-based removal and swap moves."""
        code = list(dict.fromkeys(code))
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1

        improved = True
        stall_count = 0
        max_stall = 3

        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(code)

            # Phase 1: Priority-queue-based removal
            # Build heap: (cover_count[c], c) — lower is more removable
            # A codeword is removable if all its neighbors have cover_count > 1
            # after removal, i.e., min cover_count in its ball >= 1 after decrement
            # Equivalent: min(cover_count[nb] for nb in ball(c)) >= 2 before removal
            # Actually: after removing c, each nb in ball(c) has cover_count[nb]-1.
            # We need cover_count[nb] >= 2 for all nb in ball(c), i.e., min >= 2.
            # But we also need the whole space covered, not just c's ball.
            # So: removing c is safe iff for ALL points p, cover_count[p] > 0 after removal.
            # After removal, only points in ball(c) change (decremented by 1).
            # So: safe iff min(cover_count[nb] for nb in ball(c)) >= 2.

            # Use a heap of (min_cover_in_ball, index)
            heap = []
            for i, c in enumerate(code):
                min_cov = int(np.min(cover_count[neighbors_arr[c]]))
                heapq.heappush(heap, (min_cov, i, c))

            removed_any = False
            while heap:
                if time.time() > time_limit:
                    break
                min_cov, i, c = heapq.heappop(heap)
                if i >= len(code):
                    continue
                if code[i] != c:
                    continue
                if min_cov < 2:
                    # Not removable (some neighbor would become uncovered)
                    continue
                # Try removing
                nb_arr = neighbors_arr[c]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                if np.all(cover_count > 0):
                    code.pop(i)
                    improved = True
                    removed_any = True
                    # Rebuild heap (simpler than updating)
                    heap = []
                    for j, cj in enumerate(code):
                        mc = int(np.min(cover_count[neighbors_arr[cj]]))
                        heapq.heappush(heap, (mc, j, cj))
                    break
                else:
                    cover_count[nb_arr] = old_counts

            if improved:
                stall_count = 0
                continue

            # Phase 2: Single swaps
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                c = code[i]
                nb_arr = neighbors_arr[c]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                uncovered = [nb for nb in nb_arr if cover_count[nb] == 0]
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    u0 = uncovered[0]
                    candidates = list(neighbors_list[u0])
                    random.shuffle(candidates)
                    best_new_c = None
                    best_gain = -1
                    for new_c in candidates:
                        if new_c in code:
                            continue
                        if not all(u in neighbors_set[new_c] for u in uncovered):
                            continue
                        gain = len(neighbors_set[new_c] - neighbors_set[c])
                        if gain > best_gain:
                            best_gain = gain
                            best_new_c = new_c
                    if best_new_c is not None:
                        cover_count[neighbors_arr[best_new_c]] += 1
                        code[i] = best_new_c
                        improved = True
                        break
                    cover_count[nb_arr] = old_counts

            if improved:
                stall_count = 0
                continue

            # Phase 3: 2-for-1 moves
            if len(code) >= 2:
                num_trials = min(80, len(code) * (len(code) - 1) // 2)
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    i, j = random.sample(range(len(code)), 2)
                    if i > j:
                        i, j = j, i
                    c1, c2 = code[i], code[j]
                    nb1 = neighbors_arr[c1]
                    nb2 = neighbors_arr[c2]
                    old_counts1 = cover_count[nb1].copy()
                    old_counts2 = cover_count[nb2].copy()
                    cover_count[nb1] -= 1
                    cover_count[nb2] -= 1
                    affected = set(nb1.tolist()) | set(nb2.tolist())
                    uncovered = [nb for nb in affected if cover_count[nb] == 0]
                    if not uncovered:
                        code.pop(i)
                        code.pop(j)
                        improved = True
                        break
                    else:
                        u0 = uncovered[0]
                        candidates = list(neighbors_list[u0])
                        random.shuffle(candidates)
                        found = False
                        for new_c in candidates:
                            if all(u in neighbors_set[new_c] for u in uncovered):
                                if new_c not in code:
                                    cover_count[neighbors_arr[new_c]] += 1
                                    code[i] = new_c
                                    code.pop(j)
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                        cover_count[nb1] = old_counts1
                        cover_count[nb2] = old_counts2

            if improved:
                stall_count = 0
                continue

            # Phase 4: 3-for-2 moves
            if len(code) >= 3:
                num_trials = min(40, len(code) * (len(code) - 1) * (len(code) - 2) // 6)
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    idxs = random.sample(range(len(code)), 3)
                    idxs.sort()
                    c1, c2, c3 = code[idxs[0]], code[idxs[1]], code[idxs[2]]
                    nb1 = neighbors_arr[c1]
                    nb2 = neighbors_arr[c2]
                    nb3 = neighbors_arr[c3]
                    old_counts1 = cover_count[nb1].copy()
                    old_counts2 = cover_count[nb2].copy()
                    old_counts3 = cover_count[nb3].copy()
                    cover_count[nb1] -= 1
                    cover_count[nb2] -= 1
                    cover_count[nb3] -= 1
                    affected = set(nb1.tolist()) | set(nb2.tolist()) | set(nb3.tolist())
                    uncovered = [nb for nb in affected if cover_count[nb] == 0]
                    if not uncovered:
                        for k in reversed(range(3)):
                            code.pop(idxs[k])
                        improved = True
                        break
                    else:
                        u0 = uncovered[0]
                        candidates = list(neighbors_list[u0])
                        random.shuffle(candidates)
                        found = False
                        for new_c1 in candidates:
                            if time.time() > time_limit:
                                break
                            if not all(u in neighbors_set[new_c1] for u in uncovered):
                                continue
                            if new_c1 in code:
                                continue
                            remaining_uncovered = [u for u in uncovered if u not in neighbors_set[new_c1]]
                            if not remaining_uncovered:
                                cover_count[neighbors_arr[new_c1]] += 1
                                code[idxs[0]] = new_c1
                                code.pop(idxs[2])
                                code.pop(idxs[1])
                                improved = True
                                found = True
                                break
                            u1 = remaining_uncovered[0]
                            candidates2 = list(neighbors_list[u1])
                            random.shuffle(candidates2)
                            for new_c2 in candidates2:
                                if all(u in neighbors_set[new_c2] for u in remaining_uncovered):
                                    if new_c2 not in code and new_c2 != new_c1:
                                        cover_count[neighbors_arr[new_c1]] += 1
                                        cover_count[neighbors_arr[new_c2]] += 1
                                        code[idxs[0]] = new_c1
                                        code[idxs[1]] = new_c2
                                        code.pop(idxs[2])
                                        improved = True
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                        cover_count[nb1] = old_counts1
                        cover_count[nb2] = old_counts2
                        cover_count[nb3] = old_counts3

            if not improved:
                stall_count += 1
                if stall_count >= max_stall:
                    break

        return code

    def perturb(code, k):
        """Random k-swap perturbation: remove k codewords, add k new ones."""
        code = list(code)
        if len(code) < k + 1:
            return code
        # Remove k random codewords
        to_remove = random.sample(range(len(code)), k)
        to_remove.sort(reverse=True)
        for i in to_remove:
            code.pop(i)
        # Add k new random codewords (from tetracode structure)
        all_cw = tetracode_full()
        new_cws = random.sample(all_cw, k)
        for nc in new_cws:
            if nc not in code:
                code.append(nc)
        return code

    def ils(code, time_limit):
        """Iterated Local Search."""
        code = list(dict.fromkeys(code))
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1

        best_code = list(code)
        best_size = len(code)
        cur_size = len(code)

        while time.time() < time_limit:
            # Local search
            code = fast_local_search(code, time_limit)
            cur_size = len(code)

            if cur_size < best_size:
                best_size = cur_size
                best_code = list(code)

            if best_size <= 73:
                break

            # Perturb
            k = random.choice([1, 2, 3])
            if len(code) > k + 1:
                code = perturb(code, k)
                # Ensure validity after perturbation
                code = ensure_valid(code)
                cur_size = len(code)

            # If perturbation made things worse, revert
            if cur_size > best_size + 5:
                code = list(best_code)

        return best_code

    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_code = None
    best_size = float('inf')

    def validate_and_record(code):
        nonlocal best_code, best_size
        code = list(dict.fromkeys(code))
        if len(code) >= best_size:
            return
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            return
        best_code = list(code)
        best_size = len(code)

    # Generate diverse initial codes
    inits = []
    inits.append(('tetracode_full', tetracode_full()))
    inits.append(('greedy', greedy_cover()))
    for k in [72, 75, 78]:
        inits.append((f'tetracode_partial_{k}', tetracode_partial(k)))
    for _ in range(3):
        inits.append(('random_opts', tetracode_random_options()))

    random.shuffle(inits)

    # Phase 1: ILS on each initial code
    for name, init_code in inits:
        if time.time() > deadline - 10:
            break

        init_code = ensure_valid(init_code)
        validate_and_record(init_code)

        remaining = deadline - time.time() - 10
        if remaining > 5:
            ls_time = min(remaining * 0.6, 25)
            code = fast_local_search(init_code, time.time() + ls_time)
            validate_and_record(code)

            # ILS if time remains
            remaining = deadline - time.time() - 10
            if remaining > 10 and best_size > 73:
                ils_code = ils(code, time.time() + min(remaining * 0.5, 30))
                validate_and_record(ils_code)

        if best_size <= 73:
            break

    # Phase 2: More ILS restarts from best found
    if best_code is not None and best_size > 73:
        remaining = deadline - time.time() - 8
        restarts = 0
        while remaining > 8 and restarts < 10:
            k = random.choice([1, 2, 3, 4])
            perturbed = perturb(best_code, k)
            perturbed = ensure_valid(perturbed)
            validate_and_record(perturbed)

            ls_code = fast_local_search(perturbed, time.time() + min(15, remaining - 8))
            validate_and_record(ls_code)

            remaining = deadline - time.time() - 8
            restarts += 1

            if best_size <= 73:
                break

    # Final polish
    if best_code is not None and best_size > 73 and time.time() < deadline - 5:
        final_code = fast_local_search(best_code, time.time() + min(8, deadline - time.time() - 3))
        validate_and_record(final_code)

    if best_code is None:
        best_code = tetracode_full()

    # Final verification
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best_code:
        cover_count[neighbors_arr[c]] += 1
    if np.any(cover_count == 0):
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    return [idx_to_tuple(c) for c in best_code]
