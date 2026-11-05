import numpy as np
import random
import time

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin under 150s kill

    # Specialized fast path for the actual task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        # Convert index <-> base-3 digits
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

        # Precompute balls (each word and its 14 one-position changes)
        # Use numpy to generate all positions' deltas
        q_powers = [1] * (n + 1)
        for i in range(1, n + 1):
            q_powers[i] = q_powers[i - 1] * q
        # digit value at position pos for index idx: (idx // q_powers[n-1-pos]) % q
        # delta for changing digit at pos to v: (v - old) * q_powers[n-1-pos]

        balls = np.empty((N, ball_size), dtype=np.int32)
        balls[:, 0] = np.arange(N, dtype=np.int32)
        # For each position pos, and each index, compute the two alternative values
        for pos in range(n):
            pw = q_powers[n - 1 - pos]
            # old digit
            old = (np.arange(N) // pw) % q
            # new digit choices: all v in 0..q-1 except old
            # We'll fill two columns per position (since q=3)
            col = 1 + pos * 2
            # v1 = (old + 1) % 3, v2 = (old + 2) % 3
            v1 = (old + 1) % q
            v2 = (old + 2) % q
            balls[:, col] = np.arange(N) + (v1 - old) * pw
            balls[:, col + 1] = np.arange(N) + (v2 - old) * pw

        # cover_map: for each word w, list of codewords whose ball contains w
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Ball sets for fast membership in local search
        ball_sets = [set(balls[c].tolist()) for c in range(N)]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with NumPy max-heap and random tie-breaking."""
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            cnt = np.full(N, ball_size, dtype=np.int32)
            # Heap of (cnt, code_idx) where code_idx is the index in the code list
            # We maintain a parallel array heap_code so we can map back.
            # Simpler: heap stores (cnt, c) where c is the codeword index.
            # When we pop, we check if cnt[c] still matches; if not, skip.
            heap = [(-ball_size, c) for c in range(N)]
            import heapq
            heapq.heapify(heap)
            code = []
            in_code = np.zeros(N, dtype=bool)
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            while uncovered_count > 0:
                # Pop until we find a valid candidate
                while heap:
                    neg_cnt, c = heapq.heappop(heap)
                    if not in_code[c] and cnt[c] == -neg_cnt:
                        break
                else:
                    break  # Should not happen
                code.append(c)
                in_code[c] = True
                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2]:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords (any codeword whose ball is fully covered by others)."""
            if not code:
                return code
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[balls[c]] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # if every word in ball c has cover_count >= 2, we can remove c
                    if np.all(cover_count[balls[c]] >= 2):
                        cover_count[balls[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def try_replace_two_with_one(code_list, cover_count, in_code):
            """Try to replace two codewords with one that covers all newly uncovered words."""
            n_codes = len(code_list)
            if n_codes < 2:
                return False
            # Try all pairs (bounded for time)
            for i in range(n_codes):
                for j in range(i + 1, n_codes):
                    c1 = code_list[i]
                    c2 = code_list[j]
                    # Temporarily remove both
                    uncov = set()
                    for w in balls[c1]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            uncov.add(w)
                    for w in balls[c2]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            uncov.add(w)
                    
                    if not uncov:
                        # Both can be removed entirely!
                        code_list.pop(j)
                        code_list.pop(i)
                        in_code[c1] = False
                        in_code[c2] = False
                        return True
                    
                    if len(uncov) <= 6:
                        # Find a single codeword that covers all uncov words
                        cand_pool = None
                        for w in sorted(uncov, key=lambda x: len(cover_map[x])):
                            cand_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c1 and c2 != c2]
                            if cand_pool is None:
                                cand_pool = set(cand_w)
                            else:
                                cand_pool.intersection_update(cand_w)
                            if not cand_pool:
                                break
                        
                        if cand_pool:
                            for c_add in cand_pool:
                                bs = ball_sets[c_add]
                                if all(w in bs for w in uncov):
                                    # Replace both with c_add
                                    code_list[j] = c_add
                                    code_list.pop(i)
                                    in_code[c1] = False
                                    in_code[c2] = False
                                    in_code[c_add] = True
                                    cover_count[balls[c_add]] += 1
                                    return True
                    
                    # Restore
                    for w in balls[c1]:
                        cover_count[w] += 1
                    for w in balls[c2]:
                        cover_count[w] += 1
            
            return False

        def local_search(code, seed, time_limit):
            """Local search: deletion, 1-swap, 2-for-1 replacement. Incremental cover_count."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[balls[c]] += 1

            in_code = np.zeros(N, dtype=bool)
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)

                # Deletion + 1-swap pass
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    # Try deletion
                    newly_uncovered = []
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # 1-swap: find a codeword (not in code, not c_remove) covering all newly_uncovered
                    found = -1
                    if len(newly_uncovered) <= 4:
                        # Intersect candidate sets from the most constrained word
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            cand_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(cand_w)
                            else:
                                cand_pool.intersection_update(cand_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            for c_add in cand_pool:
                                bs = ball_sets[c_add]
                                if all(w in bs for w in newly_uncovered):
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        cover_count[balls[found]] += 1
                        improved = True
                        break
                    else:
                        # restore
                        for w in balls[c_remove]:
                            cover_count[w] += 1

                # 2-for-1 replacement pass
                if time.time() < time_limit and len(code_list) >= 3:
                    if try_replace_two_with_one(code_list, cover_count, in_code):
                        improved = True

            return code_list

        best_code = None
        best_len = float('inf')

        # Phase 1: multiple greedy restarts + refine + local search
        num_restarts = 40
        for i in range(num_restarts):
            if time.time() > deadline - 15:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 10:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 8, time.time() + 5.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: ILS kicks on the best solution (remove k random, re-greedy, refine, LS)
        if best_code is not None:
            kick_iters = 25
            for kick_iter in range(kick_iters):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(1000 + kick_iter)
                # vary kick size: 3,4,5,6,7,8
                k = min([3, 4, 5, 6, 7, 8][kick_iter % 6], len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 5.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: final polish with extended local search
        if best_code is not None:
            if time.time() < deadline - 5:
                ls_code = local_search(best_code, 42, deadline - 3)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            final_code = refine(best_code)
            if time.time() < deadline - 2 and len(final_code) > 150:
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

    # Generic fallback (not expected for this task)
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
        nb_list = [idx]
        for pos in range(n):
            pw = q_powers[n - 1 - pos]
            old = (idx // pw) % q
            for new_d in range(q):
                if new_d == old:
                    continue
                nb_list.append(idx + (new_d - old) * pw)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
