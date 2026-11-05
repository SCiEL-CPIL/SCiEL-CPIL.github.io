import numpy as np
import random
import time
from itertools import combinations

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 112.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls (neighbors including self)
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Bitset representation: 2187 bits = 35 uint64s
        # Precompute ball bitsets for fast coverage computation
        NB_WORDS = 35  # ceil(2187/64)
        ball_bitsets = [None] * N
        for i in range(N):
            bs = [0] * NB_WORDS
            for w in balls[i]:
                bs[w >> 6] |= (1 << (w & 63))
            ball_bitsets[i] = bs

        # Precompute cover_map as sorted lists for faster intersection
        cover_map_sorted = [sorted(covers) for covers in cover_map]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with random tie-breaking."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            while uncovered_count > 0:
                max_val = max(cnt)
                candidates = [i for i in range(N) if cnt[i] == max_val]
                c = rng.choice(candidates)
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, 2-swap, and 3-swap moves using bitsets."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            def compute_uncovered(removed_codes):
                """Compute set of words that become uncovered when removing given codewords."""
                uncov = set()
                for rc in removed_codes:
                    for w in ball_indices[rc]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            uncov.add(w)
                return uncov

            def restore_coverage(removed_codes):
                """Restore coverage after failed removal attempt."""
                for rc in removed_codes:
                    for w in ball_indices[rc]:
                        cover_count[w] += 1

            def find_replacement(uncov_set, exclude_set):
                """Find a codeword not in exclude_set that covers all words in uncov_set."""
                if not uncov_set:
                    return -1
                # Sort by scarcity to prune early
                sorted_uncov = sorted(uncov_set, key=lambda w: len(cover_map_sorted[w]))
                cand_pool = None
                for w in sorted_uncov:
                    candidates_w = [c2 for c2 in cover_map_sorted[w] if not in_code[c2] and c2 not in exclude_set]
                    if not candidates_w:
                        return -1
                    if cand_pool is None:
                        cand_pool = set(candidates_w)
                    else:
                        cand_pool.intersection_update(candidates_w)
                        if not cand_pool:
                            return -1
                # Verify with ball_sets
                for c_add in cand_pool:
                    bs = ball_sets[c_add]
                    if all(w in bs for w in uncov_set):
                        return c_add
                return -1

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)

                # Phase 1: Deletion and 1-swap
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    newly_uncovered = compute_uncovered([c_remove])

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    if len(newly_uncovered) <= 4:
                        found = find_replacement(newly_uncovered, {c_remove})
                        if found >= 0:
                            code_list[i] = found
                            in_code[c_remove] = False
                            in_code[found] = True
                            for w in ball_indices[found]:
                                cover_count[w] += 1
                            improved = True
                            break
                    else:
                        restore_coverage([c_remove])

                # Phase 2: 2-swap (remove two, add one)
                if time.time() < time_limit and len(code_list) >= 3 and not improved:
                    n_codes = len(code_list)
                    pairs_done = 0
                    max_pairs = min(n_codes * (n_codes - 1) // 2, 200)
                    for i2 in range(n_codes):
                        if time.time() > time_limit or pairs_done >= max_pairs:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit or pairs_done >= max_pairs:
                                break
                            pairs_done += 1
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            uncov = compute_uncovered([c_rem1, c_rem2])
                            if uncov and len(uncov) <= 5:
                                found2 = find_replacement(uncov, {c_rem1, c_rem2})
                                if found2 >= 0:
                                    # Replace c_rem2 with found2, remove c_rem1
                                    code_list[j2] = found2
                                    in_code[c_rem2] = False
                                    in_code[found2] = True
                                    for w in ball_indices[found2]:
                                        cover_count[w] += 1
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    improved = True
                                    break
                            else:
                                restore_coverage([c_rem1, c_rem2])
                        if improved:
                            break

                # Phase 3: 3-swap (remove three, add one)
                if time.time() < time_limit and len(code_list) >= 4 and not improved:
                    n_codes = len(code_list)
                    num_triples = min(80, n_codes * (n_codes - 1) * (n_codes - 2) // 6)
                    triple_samples = []
                    if n_codes <= 25:
                        for combo in combinations(range(n_codes), 3):
                            triple_samples.append(combo)
                            if len(triple_samples) >= num_triples:
                                break
                    else:
                        seen = set()
                        while len(triple_samples) < num_triples:
                            t = tuple(sorted(rng.sample(range(n_codes), 3)))
                            if t not in seen:
                                seen.add(t)
                                triple_samples.append(t)

                    for (a, b, c3) in triple_samples:
                        if time.time() > time_limit:
                            break
                        c_r1, c_r2, c_r3 = code_list[a], code_list[b], code_list[c3]
                        uncov = compute_uncovered([c_r1, c_r2, c_r3])
                        if uncov and len(uncov) <= 6:
                            found3 = find_replacement(uncov, {c_r1, c_r2, c_r3})
                            if found3 >= 0:
                                # Replace c_r3 with found3, remove c_r1 and c_r2
                                code_list[c3] = found3
                                in_code[c_r3] = False
                                in_code[found3] = True
                                for w in ball_indices[found3]:
                                    cover_count[w] += 1
                                # Remove c_r1 and c_r2 from list
                                idx_r1 = code_list.index(c_r1)
                                code_list.pop(idx_r1)
                                in_code[c_r1] = False
                                idx_r2 = code_list.index(c_r2)
                                code_list.pop(idx_r2)
                                in_code[c_r2] = False
                                improved = True
                                break
                        else:
                            restore_coverage([c_r1, c_r2, c_r3])

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds, then local search each
        # refined solution. Also do kick/re-greedy.
        num_restarts = 30
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 15:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 12, time.time() + 4.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts on the best solution found so far
        if best_code is not None:
            for kick_iter in range(8):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(5, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 3.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final local search + refine
        if best_code is not None and time.time() < deadline - 8:
            ls_code = local_search(best_code, 42, deadline - 5)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            if time.time() < deadline - 3 and len(final_code) > 150:
                ls_code2 = local_search(final_code, 99, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
