import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

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
        # Precompute ball indices as numpy arrays for faster iteration
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]
        # Precompute ball as numpy array of int64 for vectorized ops
        ball_np = [np.array(b, dtype=np.int64) for b in balls]

        # Precompute for each word, the sorted list of covering codewords by rarity
        cover_map_sorted = [sorted(cm, key=lambda c: len(cover_map[c])) for cm in cover_map]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with heap-based candidate selection."""
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    break

                c = i
                code.append(c)
                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[ball_np[c]] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    if cover_count[ball_np[c]].min() > 1:
                        cover_count[ball_np[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, 2-remove-1-add, and 2-remove-2-add moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[ball_np[c]] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    ball_rm = ball_np[c_remove]

                    # Try deletion first
                    cover_count[ball_rm] -= 1
                    newly_uncovered = ball_rm[cover_count[ball_rm] == 0]

                    if len(newly_uncovered) == 0:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap: find a single codeword covering all newly_uncovered
                    found = -1
                    if len(newly_uncovered) <= 8:
                        nu_list = newly_uncovered.tolist()
                        nu_sorted = sorted(nu_list, key=lambda w: len(cover_map[w]))
                        w0 = nu_sorted[0]
                        candidates_w0 = [c2 for c2 in cover_map[w0] if not in_code[c2] and c2 != c_remove]
                        if candidates_w0:
                            for c_add in candidates_w0:
                                bs = ball_sets[c_add]
                                ok = True
                                for w in nu_sorted[1:]:
                                    if w not in bs:
                                        ok = False
                                        break
                                if ok:
                                    found = c_add
                                    break

                    if found >= 0:
                        ball_add = ball_np[found]
                        cover_count[ball_add] += 1
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        improved = True
                        break
                    else:
                        cover_count[ball_rm] += 1

                # Try 2-remove-1-add: remove two codewords, add one
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            ball_r1 = ball_np[c_rem1]
                            ball_r2 = ball_np[c_rem2]

                            cover_count[ball_r1] -= 1
                            cover_count[ball_r2] -= 1
                            uncov = np.where(cover_count == 0)[0]

                            if len(uncov) > 0 and len(uncov) <= 10:
                                uncov_list = uncov.tolist()
                                uncov_sorted = sorted(uncov_list, key=lambda x: len(cover_map[x]))
                                w0 = uncov_sorted[0]
                                candidates_w0 = [c2 for c2 in cover_map[w0]
                                                 if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                                found2 = -1
                                if candidates_w0:
                                    for c_add in candidates_w0:
                                        bs = ball_sets[c_add]
                                        ok = True
                                        for w in uncov_sorted[1:]:
                                            if w not in bs:
                                                ok = False
                                                break
                                        if ok:
                                            found2 = c_add
                                            break
                                if found2 >= 0:
                                    ball_add = ball_np[found2]
                                    cover_count[ball_add] += 1
                                    code_list[j2] = found2
                                    in_code[c_rem2] = False
                                    in_code[found2] = True
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    improved = True
                                    break
                                else:
                                    cover_count[ball_r1] += 1
                                    cover_count[ball_r2] += 1
                            else:
                                cover_count[ball_r1] += 1
                                cover_count[ball_r2] += 1
                            if improved:
                                break
                        if improved:
                            break

                # Try 2-remove-2-add: remove two codewords, add two
                if time.time() < time_limit and len(code_list) >= 4:
                    n_codes = len(code_list)
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            ball_r1 = ball_np[c_rem1]
                            ball_r2 = ball_np[c_rem2]

                            cover_count[ball_r1] -= 1
                            cover_count[ball_r2] -= 1
                            uncov = np.where(cover_count == 0)[0]

                            if len(uncov) > 0 and len(uncov) <= 12:
                                uncov_list = uncov.tolist()
                                uncov_sorted = sorted(uncov_list, key=lambda x: len(cover_map[x]))
                                w0 = uncov_sorted[0]
                                candidates_w0 = [c2 for c2 in cover_map[w0]
                                                 if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                                found_a = -1
                                found_b = -1
                                if candidates_w0:
                                    for c_add in candidates_w0:
                                        bs = ball_sets[c_add]
                                        remaining = [w for w in uncov_sorted[1:] if w not in bs]
                                        if not remaining:
                                            found_a = c_add
                                            break
                                        elif len(remaining) <= 5:
                                            # Try to find second codeword for remaining
                                            wr = remaining[0]
                                            for c_add2 in cover_map[wr]:
                                                if c_add2 == c_add or c_add2 == c_rem1 or c_add2 == c_rem2:
                                                    continue
                                                if in_code[c_add2]:
                                                    continue
                                                bs2 = ball_sets[c_add2]
                                                ok2 = True
                                                for w in remaining[1:]:
                                                    if w not in bs2:
                                                        ok2 = False
                                                        break
                                                if ok2:
                                                    found_a = c_add
                                                    found_b = c_add2
                                                    break
                                            if found_a >= 0:
                                                break

                                if found_a >= 0:
                                    cover_count[ball_np[found_a]] += 1
                                    code_list[j2] = found_a
                                    in_code[c_rem2] = False
                                    in_code[found_a] = True
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    if found_b >= 0:
                                        cover_count[ball_np[found_b]] += 1
                                        code_list.append(found_b)
                                        in_code[found_b] = True
                                    improved = True
                                    break
                                else:
                                    cover_count[ball_r1] += 1
                                    cover_count[ball_r2] += 1
                            else:
                                cover_count[ball_r1] += 1
                                cover_count[ball_r2] += 1
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds, then local search each
        num_restarts = 100
        for i in range(num_restarts):
            if time.time() > deadline - 25:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 15:
                continue
            ls_time = min(deadline - 12, time.time() + 2.5)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts on the best solution
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 8, 5, 7, 4, 6, 5, 3, 4, 5, 6, 7, 8]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 18:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 2.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final local search + refine with multiple polish passes
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(8):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 150:
                    break
                ls_code2 = local_search(final_code, 99 + polish_iter * 13, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
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
