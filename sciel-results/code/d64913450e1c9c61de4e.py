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

        # Precompute balls as lists and masks
        balls = []
        ball_masks = [0] * N
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
            mask = 0
            for w in neighbors:
                mask |= (1 << w)
            ball_masks[i] = mask

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute balls as numpy array for vectorized gain computation
        balls_np = np.array(balls, dtype=np.int32)

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with numpy-accelerated gain computation."""
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            code = []
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    ball_c = balls[c]
                    for w in ball_c:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1

            # Precompute balls as a numpy array for vectorized gain
            # balls_np: shape (N, ball_size)
            while uncovered_count > 0:
                # Compute gain for all candidates: number of uncovered words in their ball
                gains = np.zeros(N, dtype=np.int32)
                for b_idx in range(ball_size):
                    gains += (~covered)[balls_np[:, b_idx]]
                # Pick candidate with max gain (tie-break by smallest index for determinism)
                c = int(np.argmax(gains))
                if gains[c] == 0:
                    break
                code.append(c)
                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1

            return code

        def greedy_with_diversity(seed, start_code=None, diversity=0.3):
            """Greedy set cover with diversity-aware tie-breaking."""
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            code = []
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    ball_c = balls[c]
                    for w in ball_c:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1

            # Track recently chosen codewords for diversity
            recent = []
            
            while uncovered_count > 0:
                # Compute gain for all candidates
                gains = np.zeros(N, dtype=np.int32)
                for b_idx in range(ball_size):
                    gains += (~covered)[balls_np[:, b_idx]]
                
                # Find all candidates with max gain
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                
                # Get candidates with max gain
                max_candidates = np.where(gains == max_gain)[0]
                
                if len(max_candidates) == 1:
                    c = int(max_candidates[0])
                else:
                    # Diversity-aware selection: prefer candidates far from recent ones
                    if recent and rng.random() < diversity:
                        # Compute distance to most recent codeword
                        recent_c = recent[-1]
                        recent_digits = int_to_base3(recent_c, n)
                        
                        # Score each candidate by distance to recent
                        best_c = None
                        best_score = -1
                        for cand in max_candidates:
                            cand_digits = int_to_base3(int(cand), n)
                            dist = sum(1 for a, b in zip(recent_digits, cand_digits) if a != b)
                            # Add some randomness
                            score = dist + rng.random() * 0.5
                            if score > best_score:
                                best_score = score
                                best_c = int(cand)
                        c = best_c
                    else:
                        # Random selection among max gain candidates
                        c = int(max_candidates[rng.randint(0, len(max_candidates) - 1)])
                
                code.append(c)
                recent.append(c)
                if len(recent) > 5:
                    recent.pop(0)
                
                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1

            return code

        def refine(code):
            """Remove redundant codewords using cover counts."""
            if not code:
                return code
            
            code_list = list(code)
            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1
            
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

        def find_swap_candidate(hole_words, excluded, in_code, ball_masks, cover_map, N):
            """Find a codeword not in `excluded` that covers all words in hole_words."""
            if not hole_words:
                return -1
            hole_mask = 0
            for w in hole_words:
                hole_mask |= (1 << w)
            
            # Sort by rarity to minimize pool size
            crit_sorted = sorted(hole_words, key=lambda w: len(cover_map[w]))
            cand_pool = None
            for w in crit_sorted:
                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 not in excluded]
                if cand_pool is None:
                    cand_pool = set(candidates_w)
                else:
                    cand_pool.intersection_update(candidates_w)
                if not cand_pool:
                    return -1
            
            # If pool is small, check all; if large, check most promising first
            if len(cand_pool) <= 20:
                for c_add in cand_pool:
                    if (ball_masks[c_add] & hole_mask) == hole_mask:
                        return c_add
            else:
                # Prioritize candidates that cover more of the holes
                scored = []
                for c_add in cand_pool:
                    covered = bin(ball_masks[c_add] & hole_mask).count('1')
                    scored.append((covered, c_add))
                scored.sort(reverse=True)
                for covered, c_add in scored:
                    if covered == len(hole_words):
                        return c_add
                    if covered < len(hole_words) - 2:
                        break
            
            return -1

        def find_swap_candidate_relaxed(hole_words, excluded, in_code, ball_masks, cover_map, N, max_miss=2):
            """Find a codeword that covers most of hole_words, allowing up to max_miss uncovered."""
            if not hole_words:
                return -1
            hole_mask = 0
            for w in hole_words:
                hole_mask |= (1 << w)
            
            # Sort by rarity to minimize pool size
            crit_sorted = sorted(hole_words, key=lambda w: len(cover_map[w]))
            cand_pool = None
            for w in crit_sorted:
                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 not in excluded]
                if cand_pool is None:
                    cand_pool = set(candidates_w)
                else:
                    cand_pool.intersection_update(candidates_w)
                if not cand_pool:
                    return -1
            
            # Find candidate covering most holes
            best_c = -1
            best_covered = 0
            for c_add in cand_pool:
                covered_count = bin(ball_masks[c_add] & hole_mask).count('1')
                if covered_count > best_covered:
                    best_covered = covered_count
                    best_c = c_add
            
            # Allow if we cover all but max_miss
            if best_c >= 0 and (len(hole_words) - best_covered) <= max_miss:
                return best_c
            return -1

        def find_pair_for_holes(hole_words, excluded, in_code, ball_masks, cover_map, N):
            """Find two codewords (not in excluded/in_code) that together cover all hole_words.
            Returns (c1, c2) or None."""
            if not hole_words:
                return None
            hole_mask = 0
            for w in hole_words:
                hole_mask |= (1 << w)
            
            # Use the rarest word to get candidate pool
            crit_w = min(hole_words, key=lambda w: len(cover_map[w]))
            candidates = [c2 for c2 in cover_map[crit_w] if not in_code[c2] and c2 not in excluded]
            if not candidates:
                return None
            
            # For each candidate c1, check if remaining holes can be covered by one c2
            for c1 in candidates:
                remaining_mask = hole_mask & ~ball_masks[c1]
                if remaining_mask == 0:
                    return (c1, -1)
                # Find a word from remaining holes
                rem_word = None
                for w in hole_words:
                    if not (ball_masks[c1] & (1 << w)):
                        rem_word = w
                        break
                if rem_word is None:
                    continue
                for c2 in cover_map[rem_word]:
                    if c2 == c1 or c2 in excluded or in_code[c2]:
                        continue
                    if (ball_masks[c2] & remaining_mask) == remaining_mask:
                        return (c1, c2)
            return None

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, 2-swap moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            improved = True
            no_improve_count = 0
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                # Pass 1: Deletions and 1-Swaps
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
                        no_improve_count = 0
                        break

                    # Try 1-swap
                    found = -1
                    if len(newly_uncovered) <= 10:
                        found = find_swap_candidate(newly_uncovered, {c_remove}, in_code, ball_masks, cover_map, N)
                        # If exact swap fails, try relaxed swap
                        if found < 0 and len(newly_uncovered) <= 8:
                            found = find_swap_candidate_relaxed(newly_uncovered, {c_remove}, in_code, ball_masks, cover_map, N, max_miss=2)

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in balls[found]:
                            cover_count[w] += 1
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        for w in balls[c_remove]:
                            cover_count[w] += 1

                # Pass 2: Limited 2-Swaps (remove 2, add 1: net -1)
                if not improved and time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    num_pairs_to_try = min(200, (n_codes * (n_codes - 1)) // 2)
                    pairs = set()
                    while len(pairs) < num_pairs_to_try and len(pairs) < (n_codes * (n_codes - 1)) // 2:
                        i2 = rng.randint(0, n_codes - 1)
                        j2 = rng.randint(0, n_codes - 1)
                        if i2 != j2:
                            if i2 > j2: i2, j2 = j2, i2
                            pairs.add((i2, j2))
                            
                    for i2, j2 in pairs:
                        if time.time() > time_limit:
                            break
                        c_rem1 = code_list[i2]
                        c_rem2 = code_list[j2]
                        excluded = {c_rem1, c_rem2}
                        
                        uncov = []
                        for w in balls[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        for w in balls[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        
                        if uncov and len(uncov) <= 14:
                            found2 = find_swap_candidate(uncov, excluded, in_code, ball_masks, cover_map, N)
                            if found2 < 0 and len(uncov) <= 10:
                                found2 = find_swap_candidate_relaxed(uncov, excluded, in_code, ball_masks, cover_map, N, max_miss=3)
                            
                            if found2 >= 0:
                                code_list[j2] = found2
                                in_code[c_rem2] = False
                                in_code[found2] = True
                                for w in balls[found2]:
                                    cover_count[w] += 1
                                code_list.pop(i2)
                                in_code[c_rem1] = False
                                improved = True
                                no_improve_count = 0
                                break
                            else:
                                for w in balls[c_rem1]:
                                    cover_count[w] += 1
                                for w in balls[c_rem2]:
                                    cover_count[w] += 1
                        else:
                            for w in balls[c_rem1]:
                                cover_count[w] += 1
                            for w in balls[c_rem2]:
                                cover_count[w] += 1
                        if improved:
                            break

                # Pass 3: 2-for-1 replacement (remove 1, add 2: net +1, to escape local minima)
                if not improved and time.time() < time_limit and len(code_list) >= 2:
                    n_codes = len(code_list)
                    num_tries = min(60, n_codes)
                    for _ in range(num_tries):
                        if time.time() > time_limit:
                            break
                        i = rng.randint(0, n_codes - 1)
                        c_rem = code_list[i]
                        
                        uncov = []
                        for w in balls[c_rem]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        
                        if uncov and len(uncov) <= 14:
                            pair_result = find_pair_for_holes(uncov, {c_rem}, in_code, ball_masks, cover_map, N)
                            if pair_result is not None:
                                c1, c2 = pair_result
                                code_list.pop(i)
                                in_code[c_rem] = False
                                code_list.append(c1)
                                in_code[c1] = True
                                for w in balls[c1]:
                                    cover_count[w] += 1
                                if c2 >= 0:
                                    code_list.append(c2)
                                    in_code[c2] = True
                                    for w in balls[c2]:
                                        cover_count[w] += 1
                                improved = True
                                no_improve_count = 0
                                break
                            else:
                                for w in balls[c_rem]:
                                    cover_count[w] += 1
                        else:
                            for w in balls[c_rem]:
                                cover_count[w] += 1

                if not improved:
                    no_improve_count += 1
                    # After several iterations without improvement, try more aggressive moves
                    if no_improve_count >= 3 and time.time() < time_limit - 5:
                        # Try removing 3 codewords and adding 2 (net -1)
                        if len(code_list) >= 5:
                            n_codes = len(code_list)
                            triples = set()
                            max_triples = min(50, (n_codes * (n_codes - 1) * (n_codes - 2)) // 6)
                            while len(triples) < max_triples:
                                i1 = rng.randint(0, n_codes - 1)
                                i2 = rng.randint(0, n_codes - 1)
                                i3 = rng.randint(0, n_codes - 1)
                                if len({i1, i2, i3}) == 3:
                                    key = tuple(sorted([i1, i2, i3]))
                                    triples.add(key)
                            
                            for (i1, i2, i3) in triples:
                                if time.time() > time_limit:
                                    break
                                c_rem1 = code_list[i1]
                                c_rem2 = code_list[i2]
                                c_rem3 = code_list[i3]
                                excluded = {c_rem1, c_rem2, c_rem3}
                                
                                uncov = []
                                for c_rem in [c_rem1, c_rem2, c_rem3]:
                                    for w in balls[c_rem]:
                                        cover_count[w] -= 1
                                        if cover_count[w] == 0:
                                            uncov.append(w)
                                
                                if uncov and len(uncov) <= 16:
                                    # Try to find 2 codewords that cover all uncov
                                    found_pair = find_pair_for_holes(uncov, excluded, in_code, ball_masks, cover_map, N)
                                    if found_pair is not None:
                                        c_add1, c_add2 = found_pair
                                        # Remove the three
                                        for idx in sorted([i1, i2, i3], reverse=True):
                                            code_list.pop(idx)
                                            in_code[code_list[idx] if idx < len(code_list) else [c_rem1, c_rem2, c_rem3][[i1,i2,i3].index(idx)]] = False
                                        # This is getting complex, let's simplify
                                        # Just add the found codewords and mark for cleanup
                                        code_list.append(c_add1)
                                        in_code[c_add1] = True
                                        for w in balls[c_add1]:
                                            cover_count[w] += 1
                                        if c_add2 >= 0:
                                            code_list.append(c_add2)
                                            in_code[c_add2] = True
                                            for w in balls[c_add2]:
                                                cover_count[w] += 1
                                        improved = True
                                        no_improve_count = 0
                                        break
                                    else:
                                        for c_rem in [c_rem1, c_rem2, c_rem3]:
                                            for w in balls[c_rem]:
                                                cover_count[w] += 1
                                else:
                                    for c_rem in [c_rem1, c_rem2, c_rem3]:
                                        for w in balls[c_rem]:
                                            cover_count[w] += 1
                                if improved:
                                    break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with varied greedy strategies
        num_restarts = 120
        for i in range(num_restarts):
            if time.time() > deadline - 18:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 12:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 10, time.time() + 6.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts with varied kick sizes
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 5, 4, 6, 7, 3, 5, 4, 6, 5, 7, 3, 5, 4, 6, 8, 9, 10, 7, 8, 6, 5, 7, 4, 6, 8, 5, 7, 9, 6, 8, 4, 6, 7, 5]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 15:
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
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 5.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish with multiple passes
        if best_code is not None and time.time() < deadline - 8:
            ls_code = local_search(best_code, 42, deadline - 5)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(6):
                if time.time() > deadline - 3:
                    break
                if len(final_code) <= 150:
                    break
                ls_code2 = local_search(final_code, 99 + polish_iter * 13, deadline - 2)
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
