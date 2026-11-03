import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # ---- fast digit <-> int conversions (base q, little-endian position) ----
    powers = [1]
    for _ in range(n):
        powers.append(powers[-1] * q)

    def int_to_digits(num, length=n, base=q):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base=q):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # ---- precompute balls ----
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # ---- inverse index: for each word, list of centers whose ball contains it ----
    inv_ball = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls_list[c]:
            inv_ball[w].append(c)

    # ---- bitset of all centers for each word (Python int) ----
    # bit c set means center c covers word w
    word_center_bits = [0] * total_words
    for w in range(total_words):
        bits = 0
        for c in inv_ball[w]:
            bits |= (1 << c)
        word_center_bits[w] = bits

    def _time_left():
        return deadline - time.time()

    # ================= greedy set cover (fast) =================
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        # per-center gain cache
        gain = np.zeros(total_words, dtype=np.int32)
        for c in range(total_words):
            gain[c] = ball_size  # initially all words uncovered

        while _time_left() > 5.0:
            best_center = int(np.argmax(gain))
            max_gain = gain[best_center]
            if max_gain == 0:
                break
            code.append(best_center)
            b = balls[best_center]
            # update cover_count and gain for affected words
            # for each word w in ball[best_center], cover_count[w] += 1
            # gain of a center c = number of uncovered words in ball[c]
            # We maintain gain lazily: recompute for centers that share a word with best_center.
            # Efficient: for each word w in ball[best_center] that becomes newly covered,
            # decrement gain of every center covering w (i.e., centers in inv_ball[w]).
            for w in b:
                if uncovered[w]:
                    uncovered[w] = False
                    for c in inv_ball[w]:
                        gain[c] -= 1
            cover_count[b] += 1
        return code, cover_count, uncovered

    # ================= trimming =================
    def trim_code(code, cover_count):
        changed = True
        while changed and _time_left() > 3.0:
            changed = False
            to_remove = []
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                b = balls[c]
                if np.any(cover_count[b] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def get_unique_words(c, cover_count):
        b = balls[c]
        return b[cover_count[b] == 1]

    # ================= swap local search =================
    def local_search(code, cover_count, seed, max_rounds=200):
        random.seed(seed)
        code_set = set(code)
        improved = True
        swap_round = 0

        while improved and _time_left() > 8.0 and swap_round < max_rounds:
            improved = False
            swap_round += 1

            # 1) removal pass
            for ci in range(len(code) - 1, -1, -1):
                i = code[ci]
                b = balls[i]
                if not np.any(cover_count[b] == 1):
                    code.pop(ci)
                    cover_count[b] -= 1
                    code_set.discard(i)
                    improved = True
                    break
            if improved:
                continue

            # 2) swap pass
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if _time_left() < 8.0:
                    return code, cover_count
                i = code[ci]
                unique = get_unique_words(i, cover_count)
                if len(unique) == 0:
                    continue

                # candidate centers = intersection of center-bits over unique words
                cand_bits = (1 << total_words) - 1  # all ones
                for w in unique:
                    cand_bits &= word_center_bits[w]
                    if cand_bits == 0:
                        break
                if cand_bits == 0:
                    continue
                # remove current code words
                for c in code_set:
                    cand_bits &= ~(1 << c)
                if cand_bits == 0:
                    continue

                # extract candidate indices
                cand_list = []
                x = cand_bits
                while x:
                    lsb = x & (-x)
                    idx = lsb.bit_length() - 1
                    cand_list.append(idx)
                    x ^= lsb

                random.shuffle(cand_list)
                n_cand = min(200, len(cand_list))
                cand_list = cand_list[:n_cand]

                bi_set = set(balls[i].tolist())
                best_swap = None
                best_gain = -1
                for j in cand_list:
                    bj = balls[j]
                    # gain = number of uncovered words in ball[j] not in ball[i]
                    # plus we must ensure unique words of i stay covered (they are, by construction)
                    # net: removing i frees its unique words, adding j covers them; 
                    # additional benefit = uncovered words in ball[j]\ball[i]
                    gain = 0
                    for w in bj:
                        if cover_count[w] == 0 and w not in bi_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

        return code, cover_count

    # ================= structured tetracode product init =================
    def tetracode_init():
        # tetracode: length 4, q=3, radius 1, 12 words
        # known tetracode words (one representation)
        tet = [
            [0,0,0,0],[0,1,2,1],[0,2,1,2],
            [1,0,1,2],[1,1,0,1],[1,2,2,0],
            [2,0,2,0],[2,1,1,1],[2,2,0,2],
            [0,1,1,0],[1,0,2,1],[2,2,1,2],
        ]
        # product: two tetracodes -> length 8
        code = []
        seen = set()
        for a in tet:
            for b in tet:
                w = a + b
                key = tuple(w)
                if key in seen:
                    continue
                seen.add(key)
                code.append(digits_to_int(w))
        return code

    # ================= run one full attempt =================
    def run_attempt(seed, use_tet):
        random.seed(seed)
        if use_tet and _time_left() > 20.0:
            code = tetracode_init()
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in code:
                cover_count[balls[c]] += 1
            # ensure full coverage: if not covered, fill with greedy
            uncovered = np.zeros(total_words, dtype=bool)
            uncovered[:] = True
            for c in code:
                uncovered[balls[c]] = False
            if np.any(uncovered):
                # greedy fill
                gain = np.zeros(total_words, dtype=np.int32)
                for c in range(total_words):
                    gain[c] = np.sum(uncovered[balls[c]])
                while np.any(uncovered) and _time_left() > 10.0:
                    bc = int(np.argmax(gain))
                    if gain[bc] == 0:
                        break
                    code.append(bc)
                    b = balls[bc]
                    for w in b:
                        if uncovered[w]:
                            uncovered[w] = False
                            for cc in inv_ball[w]:
                                gain[cc] -= 1
                    cover_count[b] += 1
        else:
            code, cover_count, uncovered = greedy_cover()

        code, cover_count = trim_code(code, cover_count)
        code, cover_count = local_search(code, cover_count, seed, max_rounds=120)
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    # ================= main: multi-start, keep best =================
    best_code = None
    best_count = None
    best_len = 10**9

    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768,
             999, 2345, 5678, 9012, 3456, 7890, 11111, 22222, 33333, 44444]
    use_tet_flags = [True, False, True, False, False, False, True, False,
                     False, False, True, False, False, False, False, False,
                     False, False, False, False]

    for s, ut in zip(seeds, use_tet_flags):
        if _time_left() < 15.0:
            break
        code, cover_count = run_attempt(s, ut)
        # verify coverage (should be full) and length
        if len(code) < best_len:
            # quick coverage check
            covered = np.zeros(total_words, dtype=bool)
            for c in code:
                covered[balls[c]] = True
            if np.all(covered):
                best_len = len(code)
                best_code = code
                best_count = cover_count

    if best_code is None:
        # fallback: greedy
        code, cover_count, _ = greedy_cover()
        code, cover_count = trim_code(code, cover_count)
        best_code = code

    # ================= final high-precision polish =================
    # tighter: full candidate scan (no truncation), more rounds
    def polish(code, cover_count):
        code_set = set(code)
        improved = True
        rounds = 0
        while improved and _time_left() > 2.0 and rounds < 60:
            improved = False
            rounds += 1
            # removal
            for ci in range(len(code) - 1, -1, -1):
                i = code[ci]
                b = balls[i]
                if not np.any(cover_count[b] == 1):
                    code.pop(ci)
                    cover_count[b] -= 1
                    code_set.discard(i)
                    improved = True
                    break
            if improved:
                continue
            # swap with full candidate set
            for ci in range(len(code)):
                if _time_left() < 2.0:
                    return code, cover_count
                i = code[ci]
                unique = get_unique_words(i, cover_count)
                if len(unique) == 0:
                    continue
                cand_bits = (1 << total_words) - 1
                for w in unique:
                    cand_bits &= word_center_bits[w]
                    if cand_bits == 0:
                        break
                if cand_bits == 0:
                    continue
                for c in code_set:
                    cand_bits &= ~(1 << c)
                if cand_bits == 0:
                    continue
                cand_list = []
                x = cand_bits
                while x:
                    lsb = x & (-x)
                    idx = lsb.bit_length() - 1
                    cand_list.append(idx)
                    x ^= lsb
                bi_set = set(balls[i].tolist())
                best_swap = None
                best_gain = -1
                for j in cand_list:
                    bj = balls[j]
                    gain = 0
                    for w in bj:
                        if cover_count[w] == 0 and w not in bi_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = j
                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break
        return code, cover_count

    best_code, best_count = polish(list(best_code), best_count)
    best_code, best_count = trim_code(best_code, best_count)

    # final safety: ensure coverage; if broken, fall back to greedy
    covered = np.zeros(total_words, dtype=bool)
    for c in best_code:
        covered[balls[c]] = True
    if not np.all(covered):
        code, cover_count, _ = greedy_cover()
        code, cover_count = trim_code(code, cover_count)
        best_code = code

    # convert to list of digit lists
    result = []
    for c in best_code:
        result.append(int_to_digits(c))
    return result
