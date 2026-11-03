import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using structured tetracode-product initialization,
    vectorized greedy set-cover, aggressive trimming, and inverse-index-based
    targeted local search with simulated-annealing polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls
    balls = np.empty((total_words, ball_size), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # Inverse index: which centers cover each word
    centers_covering = [None] * total_words
    for i in range(total_words):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total_words):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # Structured tetracode-product initialization
    def structured_init():
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # Pure greedy set-cover fallback
    def greedy_cover(seed_offset=0):
        covered = np.zeros(total_words, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        rng = np.random.RandomState(seed_offset) if seed_offset > 0 else np.random.RandomState(0)
        
        while covered_count < total_words and time.time() < deadline - 5:
            gains = ball_size - covered[balls].sum(axis=1)
            for c in code:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[rng.randint(len(best_centers))])
            code.append(best_center)
            bp = balls[best_center]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # Choose the better starting point
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_cover()
    code_set = set(code)

    # Aggressive trimming of redundant codewords
    def trim():
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    trim()

    # Local search: targeted swap via inverse index
    def local_search(time_budget=60.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(12345)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                # Score: minimize wasted coverage after the swap
                best_swap = None
                best_waste = None
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    in_i = (cover_count[bp_i] > 0)
                    waste = int(np.sum((cover_count[bp_j] >= 2) & in_i))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    local_search(time_budget=50.0)
    trim()

    # Simulated annealing polish
    def simulated_annealing_polish(max_iters=200000):
        nonlocal code, cover_count
        code_set_local = set(code)
        candidates = list(set(range(total_words)) - code_set_local)
        cand_arr = np.array(candidates, dtype=np.int32) if candidates else np.array([], dtype=np.int32)
        n_cand = len(cand_arr)
        cc = cover_count.copy()
        code_local = code[:]
        code_set_local = set(code_local)
        
        T_start = 3.0
        T_end = 0.05
        n_sa_iters = 0
        improved = True
        rng = random.Random(99999)

        while improved and n_sa_iters < max_iters and time.time() < deadline - 5:
            improved = False
            T = T_start * (T_end / T_start) ** (n_sa_iters / max(1, max_iters))
            n_sa_iters += 1

            if len(code_local) == 0 or n_cand == 0:
                break

            move_type = rng.random()

            if move_type < 0.5:
                # Single swap
                i = rng.randint(len(code_local))
                old_i = code_local[i]
                i_ball = balls[old_i]
                only_mask = cc[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code_local.pop(i)
                    cc[i_ball] -= 1
                    code_set_local.discard(old_i)
                    candidates.append(old_i)
                    cand_arr = np.array(candidates, dtype=np.int32)
                    n_cand = len(cand_arr)
                    improved = True
                    continue

                a = rng.randint(n_cand)
                j = int(cand_arr[a])
                if j in code_set_local:
                    continue

                j_set = frozenset(balls[j].tolist())
                deficit = sum(1 for w in only_by_i if w not in j_set)

                if deficit == 0:
                    code_local[i] = j
                    code_set_local.discard(old_i)
                    code_set_local.add(j)
                    cc[i_ball] -= 1
                    cc[balls[j]] += 1
                    candidates[a] = old_i
                    improved = True
                elif deficit <= 2 and rng.random() < np.exp(-deficit / T):
                    code_local[i] = j
                    code_set_local.discard(old_i)
                    code_set_local.add(j)
                    cc[i_ball] -= 1
                    cc[balls[j]] += 1
                    candidates[a] = old_i
                    improved = True

            elif move_type < 0.7:
                # Pair swap
                if len(code_local) < 2:
                    continue
                i = rng.randint(len(code_local))
                j_idx = rng.randint(len(code_local))
                if j_idx == i:
                    continue
                a = rng.randint(n_cand)
                b = rng.randint(n_cand)
                if a == b:
                    continue
                new_a = int(cand_arr[a])
                new_b = int(cand_arr[b])
                if new_a in code_set_local or new_b in code_set_local:
                    continue

                old_i = code_local[i]
                old_j = code_local[j_idx]
                i_ball = balls[old_i]
                j_ball = balls[old_j]

                only_i = i_ball[cc[i_ball] == 1]
                only_j = j_ball[cc[j_ball] == 1]
                required = set(only_i.tolist()) | set(only_j.tolist())

                if len(required) == 0:
                    # Remove both
                    if j_idx > i:
                        code_local.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set_local.discard(old_j)
                        code_local.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set_local.discard(old_i)
                    else:
                        code_local.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set_local.discard(old_i)
                        code_local.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set_local.discard(old_j)
                    candidates.extend([old_i, old_j])
                    cand_arr = np.array(candidates, dtype=np.int32)
                    n_cand = len(cand_arr)
                    improved = True
                    continue

                na_set = frozenset(balls[new_a].tolist())
                nb_set = frozenset(balls[new_b].tolist())
                deficit = sum(1 for w in required if w not in na_set and w not in nb_set)

                if deficit == 0:
                    code_local[i] = new_a
                    code_local[j_idx] = new_b
                    code_set_local.discard(old_i)
                    code_set_local.discard(old_j)
                    code_set_local.add(new_a)
                    code_set_local.add(new_b)
                    cc[i_ball] -= 1
                    cc[j_ball] -= 1
                    cc[balls[new_a]] += 1
                    cc[balls[new_b]] += 1
                    candidates[a] = old_i
                    candidates[b] = old_j
                    improved = True
                elif deficit <= 2 and rng.random() < np.exp(-deficit / T):
                    code_local[i] = new_a
                    code_local[j_idx] = new_b
                    code_set_local.discard(old_i)
                    code_set_local.discard(old_j)
                    code_set_local.add(new_a)
                    code_set_local.add(new_b)
                    cc[i_ball] -= 1
                    cc[j_ball] -= 1
                    cc[balls[new_a]] += 1
                    cc[balls[new_b]] += 1
                    candidates[a] = old_i
                    candidates[b] = old_j
                    improved = True

            else:
                # Try removal
                i = rng.randint(len(code_local))
                old_i = code_local[i]
                i_ball = balls[old_i]
                if np.any(cc[i_ball] == 1):
                    continue
                code_local.pop(i)
                cc[i_ball] -= 1
                code_set_local.discard(old_i)
                candidates.append(old_i)
                cand_arr = np.array(candidates, dtype=np.int32)
                n_cand = len(cand_arr)
                improved = True

        # Final trim
        code_local, cc = trim_local(code_local, cc)
        return code_local, cc

    def trim_local(code_local, cc):
        changed = True
        code_set_local = set(code_local)
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(code_local):
                if (cc[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code_local[idx]
                    code_local.pop(idx)
                    cc[balls[c]] -= 1
                    code_set_local.discard(c)
                changed = True
        return code_local, cc

    code, cover_count = simulated_annealing_polish(max_iters=100000)
    
    # Final targeted local search
    local_search(time_budget=30.0)
    trim()

    # Multi-start: try a few more greedy starts if time permits
    best_code = code[:]
    best_count = len(code)
    
    if time.time() < deadline - 30:
        for start_i in range(3):
            if time.time() > deadline - 20:
                break
            code2, cc2 = greedy_cover(seed_offset=start_i * 1000 + 1)
            code2, cc2 = trim_local(code2, cc2)
            
            # Quick local search
            code_set2 = set(code2)
            end_time = time.time() + 10
            improved = True
            while improved and time.time() < end_time:
                improved = False
                indices = list(range(len(code2)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > end_time:
                        break
                    i = code2[idx]
                    bp_i = balls[i]
                    unique_pts = bp_i[cc2[bp_i] == 1]
                    if len(unique_pts) == 0:
                        cc2[bp_i] -= 1
                        code2.pop(idx)
                        code_set2.discard(i)
                        improved = True
                        break
                    candidate_set = None
                    for w in unique_pts:
                        ccc = centers_covering[w]
                        if candidate_set is None:
                            candidate_set = set(ccc)
                        else:
                            candidate_set &= ccc
                        if not candidate_set:
                            break
                    if not candidate_set:
                        continue
                    candidates = [c for c in candidate_set if c not in code_set2]
                    if not candidates:
                        continue
                    best_swap = None
                    best_waste = None
                    for j in candidates:
                        bp_j = balls[j]
                        in_i = (cc2[bp_i] > 0)
                        waste = int(np.sum((cc2[bp_j] >= 2) & in_i))
                        if best_waste is None or waste < best_waste:
                            best_waste = waste
                            best_swap = j
                    if best_swap is not None:
                        j = best_swap
                        cc2[balls[i]] -= 1
                        cc2[balls[j]] += 1
                        code2[idx] = j
                        code_set2.discard(i)
                        code_set2.add(j)
                        improved = True
                        break
            
            code2, cc2 = trim_local(code2, cc2)
            if len(code2) < best_count:
                best_count = len(code2)
                best_code = code2[:]

    # Final polish on best solution
    if best_code and time.time() < deadline - 10:
        cover_count_final = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count_final[balls[c]] += 1
        code_set_final = set(best_code)
        
        end_time = time.time() + (deadline - 10 - time.time()) * 0.8
        improved = True
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(best_code)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = best_code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count_final[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count_final[bp_i] -= 1
                    best_code.pop(idx)
                    code_set_final.discard(i)
                    improved = True
                    break
                candidate_set = None
                for w in unique_pts:
                    ccc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(ccc)
                    else:
                        candidate_set &= ccc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set_final]
                if not candidates:
                    continue
                best_swap = None
                best_waste = None
                for j in candidates:
                    bp_j = balls[j]
                    in_i = (cover_count_final[bp_i] > 0)
                    waste = int(np.sum((cover_count_final[bp_j] >= 2) & in_i))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count_final[balls[i]] -= 1
                    cover_count_final[balls[j]] += 1
                    best_code[idx] = j
                    code_set_final.discard(i)
                    code_set_final.add(j)
                    improved = True
                    break
        
        # Final trim
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(best_code):
                if (cover_count_final[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = best_code[idx]
                    best_code.pop(idx)
                    cover_count_final[balls[c]] -= 1
                    code_set_final.discard(c)
                changed = True

    if not best_code:
        best_code = code[:]

    result = [int_to_digits(c, n, q) for c in best_code]
    return result
