import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 112.0  # Safety margin for 120s limit

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
        # Precompute ball as numpy array for vectorized ops
        ball_np = [np.array(b, dtype=np.int64) for b in balls]

        # ============ ALGEBRAIC STRUCTURE: Hamming Code [7,4,3] ============
        # Generator matrix G (4x7) over GF(3):
        # The ternary Hamming code [7,4,3] has 81 codewords (perfect code).
        # Its 9 cosets partition the 2187 words into 9 groups of 243.
        # Each coset is a "shift" of the perfect code.
        # We use coset representatives to guide initialization.

        # Parity check matrix H (3x7) for ternary Hamming code:
        # Columns are all distinct nonzero vectors in GF(3)^3 up to scalar
        # We use the standard construction: columns are (1,0,0), (0,1,0), (0,0,1),
        # (1,1,0), (1,0,1), (0,1,1), (1,1,1)
        H_cols = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        H = np.array(H_cols, dtype=int).T  # 7x3

        # Compute syndrome for each word (syndrome = H @ w mod 3)
        # Syndrome has 9 possible values (3^3)
        syndromes = np.zeros(N, dtype=np.int32)
        for i in range(N):
            digits = int_to_base3(i, n)
            s = (H @ np.array(digits, dtype=int)) % 3
            syndromes[i] = s[0] * 9 + s[1] * 3 + s[2]

        # Group words by syndrome (coset)
        cosets = {s: [] for s in range(9)}
        for i in range(N):
            cosets[syndromes[i]].append(i)

        # For each coset, the words form a "shifted" perfect code structure.
        # Within a coset, any two words differ in at least 3 positions (if they
        # are in the same "sub-coset" of the Hamming code). This gives us a
        # natural way to pick well-spread codewords.

        # ============ GREEDY WITH COSET AWARENESS ============
        def greedy_cover_coset(seed, start_code=None):
            """Greedy set cover that prioritizes words in under-represented cosets."""
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            in_code = [False] * N
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

            # Build heap with coset-aware priority
            # Priority: (-cnt[i], -coset_bonus[i], i)
            # coset_bonus: how many uncovered words are in the same coset as i
            heap = []
            for i in range(N):
                if not in_code[i]:
                    # Bonus: prefer codewords whose coset has more uncovered words
                    coset_id = syndromes[i]
                    coset_words = cosets[coset_id]
                    uncovered_in_coset = sum(1 for w in coset_words if not covered[w])
                    priority = (-cnt[i], -uncovered_in_coset, i)
                    heap.append(priority)
            heapq.heapify(heap)

            while uncovered_count > 0:
                while heap:
                    neg_cnt, neg_bonus, i = heapq.heappop(heap)
                    if cnt[i] == -neg_cnt and not in_code[i]:
                        break
                else:
                    break

                c = i
                code.append(c)
                in_code[c] = True

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2]:
                                # Recompute bonus for c2
                                coset_id2 = syndromes[c2]
                                coset_words2 = cosets[coset_id2]
                                uncovered_in_coset2 = sum(1 for w2 in coset_words2 if not covered[w2])
                                heapq.heappush(heap, (-cnt[c2], -uncovered_in_coset2, c2))

            return code

        def greedy_cover(seed, start_code=None):
            """Standard greedy set cover with heap-based candidate selection."""
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            in_code = [False] * N
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

            heap = []
            for i in range(N):
                if not in_code[i]:
                    heap.append((-cnt[i], i))
            heapq.heapify(heap)

            while uncovered_count > 0:
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not in_code[i]:
                        break
                else:
                    break

                c = i
                code.append(c)
                in_code[c] = True

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2]:
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
            """Local search with deletion, 1-swap, 2-remove-1-add moves."""
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

            return code_list

        # ============ ALGEBRAIC INITIALIZATION ============
        def hamming_init(seed):
            """Initialize using Hamming code coset structure.
            
            The ternary Hamming code [7,4,3] has 81 codewords forming a perfect
            covering (each word is within distance 1 of exactly one codeword).
            Its 9 cosets partition the space. We greedily select codewords
            within each coset to build a near-optimal covering.
            """
            rng = random.Random(seed)
            covered = [False] * N
            code = []
            uncovered_count = N

            # Process cosets in a randomized order
            coset_order = list(range(9))
            rng.shuffle(coset_order)

            for coset_id in coset_order:
                if time.time() > deadline - 20:
                    break
                coset_words = cosets[coset_id]
                # Within this coset, greedily pick codewords that maximize
                # coverage of still-uncovered words
                remaining = [w for w in coset_words if not covered[w]]
                while remaining and uncovered_count > 0:
                    # Pick the codeword in remaining that covers the most uncovered words
                    best_w = -1
                    best_gain = 0
                    for w in remaining:
                        gain = sum(1 for x in balls[w] if not covered[x])
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                    if best_w < 0 or best_gain <= 0:
                        break
                    code.append(best_w)
                    for x in balls[best_w]:
                        if not covered[x]:
                            covered[x] = True
                            uncovered_count -= 1
                    # Remove covered words from remaining
                    remaining = [w for w in remaining if not covered[w]]

            # Fill in any remaining gaps with standard greedy
            if uncovered_count > 0:
                code = greedy_cover(seed + 100, start_code=code)

            return code

        best_code = None
        best_len = float('inf')

        # Phase 1: Hamming-based initialization + local search
        num_hamming_restarts = 15
        for i in range(num_hamming_restarts):
            if time.time() > deadline - 30:
                break
            code = hamming_init(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 20:
                continue
            ls_time = min(deadline - 15, time.time() + 3.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: Coset-aware greedy + local search
        num_coset_restarts = 20
        for i in range(num_coset_restarts):
            if time.time() > deadline - 25:
                break
            code = greedy_cover_coset(i + 50)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 15:
                continue
            ls_time = min(deadline - 12, time.time() + 2.5)
            ls_code = local_search(refined, (i + 50) * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 3: Standard greedy + local search (for diversity)
        num_std_restarts = 25
        for i in range(num_std_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i + 200)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 12:
                continue
            ls_time = min(deadline - 10, time.time() + 2.0)
            ls_code = local_search(refined, (i + 200) * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts on the best solution
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 8, 5, 7, 4, 6, 5, 3, 4, 5, 6, 7, 8]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                # Alternate between coset-aware and standard greedy for kick restarts
                if kick_iter % 2 == 0:
                    re_code = greedy_cover_coset(2000 + kick_iter, start_code=remaining)
                else:
                    re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 2.0))
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
            for polish_iter in range(10):
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
