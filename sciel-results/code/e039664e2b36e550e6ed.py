import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute balls
    def build_balls():
        balls = np.empty((total, ball_size), dtype=np.int32)
        for w in range(total):
            digits = []
            tmp = w
            for i in range(n):
                digits.append(tmp % q)
                tmp //= q
            digits = digits[::-1]
            ball = [w]
            for pos in range(n):
                orig = digits[pos]
                for nv in range(q):
                    if nv != orig:
                        new_word = digits[:]
                        new_word[pos] = nv
                        val = 0
                        for d in new_word:
                            val = val * q + d
                        ball.append(val)
            balls[w] = ball
        return balls

    balls = build_balls()

    # Inverse index
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # Vectorized gain-based greedy
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        gain = np.full(total, ball_size, dtype=np.int32)
        code = []
        cover_count = np.zeros(total, dtype=np.int32)

        while np.any(~covered):
            if time.time() > deadline - 5:
                break
            best_word = int(np.argmax(gain))
            if gain[best_word] == 0:
                break
            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                cover_count[bp] += 1
                # Update gains: for each newly covered word v, all centers that cover v lose 1 gain
                newly_idx = bp[newly]
                for v in newly_idx:
                    for c in centers_covering[v]:
                        gain[c] -= 1

        return code, cover_count

    code, cover_count = greedy_cover()
    if not code:
        code = [0]
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[0]] += 1

    # Trim redundant codewords
    def trim(code, cover_count):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if time.time() > deadline - 10:
                    break
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    code, cover_count = trim(code, cover_count)

    # Local search: swap to minimize unique points of new codeword
    def local_search(code, cover_count, time_budget=60):
        code_set = set(code)
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
                unique_mask = cover_count[bp_i] == 1
                unique_pts = bp_i[unique_mask]

                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates covering all unique_pts
                candidate_set = None
                for w in unique_pts:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue

                in_bp_i = np.zeros(total, dtype=np.int8)
                in_bp_i[bp_i] = 1

                best_cand = None
                best_score = float('inf')
                for cand in candidates:
                    bp_cand = balls[cand]
                    score = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                    if score < best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is not None:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    bp_new = balls[best_cand]
                    cover_count[bp_new] += 1
                    code.append(best_cand)
                    code_set.add(best_cand)
                    improved = True

        return code, cover_count

    # Iterated Local Search
    best_code = code[:]
    best_count = cover_count.copy()
    best_size = len(code)
    rng = random.Random(42)
    ils_end = time.time() + 90.0

    while time.time() < ils_end:
        # Local search
        code, cover_count = local_search(code, cover_count, time_budget=40)
        code, cover_count = trim(code, cover_count)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            best_count = cover_count.copy()
            code = best_code[:]
            cover_count = best_count.copy()
        elif time.time() > ils_end - 10:
            break

        # Perturbation: random swaps to escape local minimum
        if len(code) > 2:
            num_pert = rng.randint(1, 3)
            for _ in range(num_pert):
                if len(code) < 2:
                    break
                idx = rng.randint(0, len(code) - 1)
                i = code[idx]
                bp_i = balls[i]
                # Find a random candidate to swap with
                # Pick a random point in bp_i, find a center covering it that's not in code
                rand_pt = bp_i[rng.randint(0, ball_size - 1)]
                cands = [c for c in centers_covering[rand_pt] if c not in set(code)]
                if cands:
                    cand = cands[rng.randint(0, len(cands) - 1)]
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    bp_new = balls[cand]
                    cover_count[bp_new] += 1
                    code.append(cand)
            # Re-trim after perturbation
            code, cover_count = trim(code, cover_count)

    code, cover_count = best_code, best_count

    # Final polish
    code, cover_count = local_search(code, cover_count, time_budget=15)
    code, cover_count = trim(code, cover_count)

    # Convert to word lists
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [to_word(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
