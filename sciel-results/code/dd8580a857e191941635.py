import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Covering code K_3(8,1) using:
      1. Structured tetracode-product initialization (inspiration 1)
      2. Vectorized greedy set-cover extension
      3. Redundancy trimming
      4. Inverse-index-based targeted swap local search (inspiration 1)
      5. Multi-start with different seeds and polish passes
    """
    start_time = time.time()
    deadline = start_time + 145.0

    total = q ** n
    ball_size = 1 + n * (q - 1)  # radius = 1

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

    # ---------- Precompute balls ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
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

    # ---------- Inverse index: which centers cover each word ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Structured tetracode-product initialization ----------
    def structured_init(seed_offset=0):
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
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        # Greedy extension of the structured seed
        uncovered = (cover_count == 0)
        rng = np.random.RandomState(seed_offset)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[rng.randint(len(best_centers))])
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # ---------- Pure greedy set-cover fallback ----------
    def greedy_cover(seed_offset=0):
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        rng = np.random.RandomState(seed_offset)
        while covered_count < total and time.time() < deadline - 5:
            gains = ball_size - covered[balls].sum(axis=1)
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_word = int(best_centers[0])
            else:
                best_word = int(best_centers[rng.randint(len(best_centers))])
            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # ---------- Aggressive trimming of redundant codewords ----------
    def trim(code, cover_count, code_set):
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
        return code, cover_count

    # ---------- Local search: targeted swap via inverse index ----------
    def local_search(code, cover_count, code_set, time_budget=60.0, seed=12345):
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(seed)
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
                        return code, cover_count, code_set
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
        return code, cover_count, code_set

    # ---------- Simulated annealing polish ----------
    def simulated_annealing_polish(code, cover_count, code_set, max_iters=100000, seed=999):
        rng = random.Random(seed)
        T_start = 3.0
        T_end = 0.05
        n_sa_iters = 0
        improved = True

        while improved and n_sa_iters < max_iters and time.time() < deadline - 5:
            improved = False
            T = T_start * (T_end / T_start) ** (n_sa_iters / max(1, max_iters))
            n_sa_iters += 1

            if len(code) == 0:
                break

            move_type = rng.random()

            if move_type < 0.5:
                # Single swap
                i = rng.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(i)
                    cover_count[i_ball] -= 1
                    code_set.discard(old_i)
                    improved = True
                    continue

                # Find candidates via intersection
                candidate_set = None
                for w in only_by_i:
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

                j = candidates[rng.randint(len(candidates))]
                j_set = set(balls[j].tolist())
                deficit = sum(1 for w in only_by_i if w not in j_set)

                if deficit == 0:
                    code[i] = j
                    code_set.discard(old_i)
                    code_set.add(j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                elif deficit <= 2 and rng.random() < np.exp(-deficit / T):
                    code[i] = j
                    code_set.discard(old_i)
                    code_set.add(j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[j]] += 1
                    improved = True

            else:
                # Try removal
                i = rng.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                if np.any(cover_count[i_ball] == 1):
                    continue
                code.pop(i)
                cover_count[i_ball] -= 1
                code_set.discard(old_i)
                improved = True

        return code, cover_count, code_set

    # ---------- Multi-start with different random seeds ----------
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 4
    for start_i in range(n_starts):
        if time.time() > deadline - 45:
            break

        # Use structured init for first two starts, greedy for others
        if start_i < 2:
            code, cover_count = structured_init(seed_offset=start_i * 1000 + 1)
        else:
            code, cover_count = greedy_cover(seed_offset=start_i * 1000 + 1)

        code_set = set(code)
        code, cover_count = trim(code, cover_count, code_set)
        code, cover_count, code_set = local_search(code, cover_count, code_set, time_budget=50.0, seed=12345 + start_i)
        code, cover_count, code_set = simulated_annealing_polish(code, cover_count, code_set, max_iters=50000, seed=999 + start_i)
        code, cover_count = trim(code, cover_count, code_set)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]
            best_code_set = code_set.copy()
            best_cover_count = cover_count.copy()

    # Final polish on best solution
    if best_code is not None:
        code, cover_count, code_set = local_search(best_code, best_cover_count, best_code_set, time_budget=40.0, seed=77777)
        code, cover_count, code_set = simulated_annealing_polish(code, cover_count, code_set, max_iters=80000, seed=88888)
        code, cover_count = trim(code, cover_count, code_set)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
