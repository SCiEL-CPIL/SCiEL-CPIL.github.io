import time
import numpy as np
import random
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17
    words_per_int = 64  # 64 bits per python int -> 6561/64 = 103 ints

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls as integer lists
    balls = np.zeros((total, num_neighbors), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Bit-packed ball membership: ballset[v][k] is a 64-bit chunk; bit j in chunk k
    # is 1 iff word (k*64 + j) is in the radius-1 ball of word v.
    num_chunks = (total + words_per_int - 1) // words_per_int  # 103
    ballset = np.zeros((total, num_chunks), dtype=np.uint64)
    for w in range(total):
        for nb in balls[w]:
            chunk = nb >> 6
            bit = nb & 63
            ballset[w, chunk] |= np.uint64(1) << np.uint64(bit)

    # Precompute tetracode integer words
    tetracode_ints = []
    for a in range(3):
        for b in range(3):
            word = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            w_int = 0
            for d in word:
                w_int = w_int * 3 + d
            tetracode_ints.append(w_int)

    def _bitset_popcount(b):
        # b is a list/array of uint64
        return int(np.bitwise_count(b).sum()) if hasattr(np, 'bitwise_count') else \
            sum(bin(int(x)).count('1') for x in b)

    def _covers_all(cand, words):
        # True if ball of cand contains every word in `words` (words: list of ints)
        b = ballset[cand]
        for w in words:
            chunk = w >> 6
            bit = w & 63
            if (int(b[chunk]) >> bit) & 1 == 0:
                return False
        return True

    def _swap_candidates(c, U_c):
        # Return array of codewords c' (c' != c) whose ball covers all of U_c.
        if len(U_c) == 0:
            return np.array([], dtype=np.int32)
        # Intersect bitsets of all words in U_c
        mask = np.array(ballset[U_c[0]], dtype=np.uint64)
        for w in U_c[1:]:
            mask &= np.array(ballset[w], dtype=np.uint64)
        # Expand mask to word indices
        words = np.flatnonzero(np.concatenate([
            np.array([int(x) for x in mask], dtype=np.uint64)
        ]))
        # Build index list from chunks
        idx = []
        for k in range(num_chunks):
            v = int(mask[k])
            if v == 0:
                continue
            base = k << 6
            while v:
                low = v & (-v)
                bit = (low.bit_length() - 1)
                idx.append(base + bit)
                v ^= low
        arr = np.array(idx, dtype=np.int32)
        arr = arr[arr != c]
        return arr

    def make_tetracode_init(seed_val):
        random.seed(seed_val)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        for tc in tetracode_ints:
            for tail in itertools.product(range(3), repeat=4):
                w_int = tc
                for d in tail:
                    w_int = w_int * 3 + d
                if uncovered[w_int]:
                    selected.append(w_int)
                    cover_count[balls[w_int]] += 1
                    uncovered[balls[w_int]] = False
        while np.any(uncovered):
            if time.time() > deadline - 15.0:
                break
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = (cover_count == 0)
        return selected, cover_count

    def make_random_greedy_init(seed_val):
        random.seed(seed_val)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        while np.any(uncovered):
            if time.time() > deadline - 15.0:
                break
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = (cover_count == 0)
        return selected, cover_count

    def remove_redundant(selected, cover_count, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5.0
        if not selected:
            return selected, cover_count
        current_set = set(selected)
        cover_count = cover_count.copy()
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            min_count = np.min(cover_count)
            if min_count >= 2:
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls[c_to_remove]] -= 1
                changed = True
                continue
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if np.sum(cover_count[balls[c]] == 1) == 0:
                    current_set.remove(c)
                    cover_count[balls[c]] -= 1
                    changed = True
                    break
        return list(current_set), cover_count

    def local_search(selected, cover_count, time_limit=None, do_two_exchange=True):
        if time_limit is None:
            time_limit = deadline - 5.0
        current_set = set(selected)
        cover_count = cover_count.copy()
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if time.time() > time_limit:
                    break
                c_ball = balls[c]
                U_c = c_ball[cover_count[c_ball] == 1]
                if len(U_c) == 0:
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break
                valid = _swap_candidates(c, U_c)
                if len(valid) == 0:
                    continue
                # Try up to a few candidates; pick one that lets us remove another word
                tried = set()
                for _ in range(min(4, len(valid))):
                    c_prime = int(valid[random.randint(0, len(valid) - 1)])
                    if c_prime in tried:
                        continue
                    tried.add(c_prime)
                    new_cover = cover_count.copy()
                    new_cover[c_ball] -= 1
                    new_cover[balls[c_prime]] += 1
                    can_remove_more = False
                    for other in current_set:
                        if other == c:
                            continue
                        if np.all(new_cover[balls[other]] > 1):
                            can_remove_more = True
                            break
                    if can_remove_more:
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        current_set.add(c_prime)
                        cover_count[balls[c_prime]] += 1
                        improved = True
                        break
                if improved:
                    break
            # Optional 2-exchange: remove two codewords, add one that covers the union of their critical words
            if do_two_exchange and not improved and time.time() < time_limit:
                cand_list = list(current_set)
                if len(cand_list) >= 2:
                    random.shuffle(cand_list)
                    for i in range(min(10, len(cand_list) - 1)):
                        if time.time() > time_limit:
                            break
                        c1 = cand_list[i]
                        c2 = cand_list[i + 1]
                        b1 = balls[c1]
                        b2 = balls[c2]
                        U1 = b1[cover_count[b1] == 1]
                        U2 = b2[cover_count[b2] == 1]
                        # union of critical words
                        union_words = np.unique(np.concatenate([U1, U2])) if len(U1) + len(U2) > 0 else np.array([], dtype=np.int32)
                        if len(union_words) == 0:
                            # both redundant individually? remove both
                            current_set.remove(c1)
                            current_set.remove(c2)
                            cover_count[b1] -= 1
                            cover_count[b2] -= 1
                            improved = True
                            break
                        valid = _swap_candidates(-1, union_words)  # any word covering all union words
                        if len(valid) == 0:
                            continue
                        # ensure the new word is not c1 or c2
                        valid = valid[(valid != c1) & (valid != c2)]
                        if len(valid) == 0:
                            continue
                        c_prime = int(valid[random.randint(0, len(valid) - 1)])
                        new_cover = cover_count.copy()
                        new_cover[b1] -= 1
                        new_cover[b2] -= 1
                        new_cover[balls[c_prime]] += 1
                        if np.min(new_cover) >= 0:
                            current_set.remove(c1)
                            current_set.remove(c2)
                            cover_count[b1] -= 1
                            cover_count[b2] -= 1
                            current_set.add(c_prime)
                            cover_count[balls[c_prime]] += 1
                            improved = True
                            break
        return list(current_set), cover_count

    def polish(selected, cover_count, time_limit):
        if time.time() > time_limit:
            return selected, cover_count
        current_set = set(selected)
        cover_count = cover_count.copy()
        for _ in range(20):
            if time.time() > time_limit:
                break
            # removal pass
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                min_count = np.min(cover_count)
                if min_count >= 2:
                    c_to_remove = next(iter(current_set))
                    current_set.remove(c_to_remove)
                    cover_count[balls[c_to_remove]] -= 1
                    changed = True
                    continue
                candidates = list(current_set)
                random.shuffle(candidates)
                for c in candidates:
                    if np.sum(cover_count[balls[c]] == 1) == 0:
                        current_set.remove(c)
                        cover_count[balls[c]] -= 1
                        changed = True
                        break
            # swap pass (bit-parallel)
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                candidates = list(current_set)
                random.shuffle(candidates)
                for c in candidates:
                    if time.time() > time_limit:
                        break
                    c_ball = balls[c]
                    U_c = c_ball[cover_count[c_ball] == 1]
                    if len(U_c) == 0:
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        improved = True
                        break
                    valid = _swap_candidates(c, U_c)
                    if len(valid) == 0:
                        continue
                    best_c_prime = None
                    best_removals = 0
                    for c_prime in valid[:25]:
                        c_prime = int(c_prime)
                        new_cover = cover_count.copy()
                        new_cover[c_ball] -= 1
                        new_cover[balls[c_prime]] += 1
                        removals = 0
                        for other in current_set:
                            if other != c and np.all(new_cover[balls[other]] > 1):
                                removals += 1
                        if removals > best_removals:
                            best_removals = removals
                            best_c_prime = c_prime
                    if best_c_prime is not None and best_removals > 0:
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        current_set.add(best_c_prime)
                        cover_count[balls[best_c_prime]] += 1
                        improved = True
                        break
            if not improved and not changed:
                break
        return list(current_set), cover_count

    best_size = total
    best_set = None
    best_cover = None

    num_restarts = 60
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        if i % 3 == 0:
            selected, cover_count = make_tetracode_init(i * 7 + 1)
        elif i % 3 == 1:
            selected, cover_count = make_random_greedy_init(i * 13 + 2)
        else:
            selected, cover_count = make_tetracode_init(i * 19 + 3)
            if time.time() < deadline - 10.0:
                uncovered = (cover_count == 0)
                while np.any(uncovered) and time.time() < deadline - 10.0:
                    counts = uncovered[balls].sum(axis=1)
                    max_count = np.max(counts)
                    if max_count == 0:
                        break
                    candidates = np.where(counts == max_count)[0]
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                    selected.append(best_idx)
                    cover_count[balls[best_idx]] += 1
                    uncovered = (cover_count == 0)

        if len(selected) >= best_size:
            continue

        selected, cover_count = remove_redundant(selected, cover_count)
        if len(selected) >= best_size:
            continue

        selected, cover_count = local_search(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected
            best_cover = cover_count.copy()
            if time.time() < deadline - 10.0:
                best_set, best_cover = polish(best_set, best_cover, deadline - 5.0)
                best_size = len(best_set)

    if best_set is not None and time.time() < deadline - 3.0:
        best_set, best_cover = polish(best_set, best_cover, deadline - 2.0)
        best_size = len(best_set)

    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
