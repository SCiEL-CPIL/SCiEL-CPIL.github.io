import time
import numpy as np
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

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

    # ---------- Precompute balls as a flat numpy array ----------
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
    # Use lists for fast intersection
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)

    # ---------- Structured tetracode-product initialization ----------
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
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        return code_ints, cover_count

    # ---------- Heap-based greedy set-cover extension ----------
    def greedy_extend(code_ints, cover_count):
        uncovered = (cover_count == 0)
        # Build a max-heap of (-gain, center)
        # gain = number of uncovered words in ball[center]
        heap = []
        for c in range(total):
            g = int(np.sum(uncovered[balls[c]]))
            if g > 0:
                heapq.heappush(heap, (-g, c))

        in_code = set(code_ints)

        while heap and time.time() < deadline - 10:
            neg_gain, c = heapq.heappop(heap)
            gain = -neg_gain

            # Recompute actual gain (lazy deletion)
            actual_gain = int(np.sum(uncovered[balls[c]]))
            if actual_gain == 0 or c in in_code:
                continue

            # Add this center
            code_ints.append(c)
            in_code.add(c)
            bp = balls[c]
            cover_count[bp] += 1
            newly_uncovered = uncovered[bp]
            uncovered[bp] = False

            # Update heap for affected centers
            affected = set()
            for w in bp:
                for cc in centers_covering[w]:
                    if cc not in in_code:
                        affected.add(cc)

            for cc in affected:
                new_gain = int(np.sum(uncovered[balls[cc]]))
                if new_gain > 0:
                    heapq.heappush(heap, (-new_gain, cc))

        return code_ints, cover_count

    # ---------- Choose starting point ----------
    code, cover_count = structured_init()
    code, cover_count = greedy_extend(code, cover_count)
    code_set = set(code)

    # ---------- Aggressive trimming of redundant codewords ----------
    def trim():
        changed = True
        while changed and time.time() < deadline - 15:
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

    # ---------- Local search: targeted swap via inverse index ----------
    def local_search(time_budget=100.0):
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
                    # Can remove this codeword
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
                        candidate_set &= set(cc)
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
                # Limit candidates for speed
                rng.shuffle(candidates)
                candidates = candidates[:50]

                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    # Waste: words in bp_j that are already covered by 2+ centers
                    # AND are also in bp_i (so they remain covered after swap)
                    waste = int(np.sum((cover_count[bp_j] >= 2)))
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

    # Run local search with time budget
    local_search(time_budget=100.0)
    trim()

    # Final polish
    if time.time() < deadline - 5:
        local_search(time_budget=min(30.0, deadline - time.time() - 5))
        trim()

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    return result
