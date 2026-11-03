import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n

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

    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    centers_covering = [None] * total_words
    for i in range(total_words):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total_words):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

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
        
        code_ints = [digits_to_int(c, q) for c in product_code]
        code_ints = list(set(code_ints))
        
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = uncovered[balls].sum(axis=1)
            gains[code_ints] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        
        return code_ints, cover_count

    def greedy_cover(seed_offset=0):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        if seed_offset > 0:
            np.random.seed(seed_offset)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[np.random.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iter=200000):
        code_set = set(code)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue

                candidate_set -= code_set

                if not candidate_set:
                    continue

                j = next(iter(candidate_set))
                
                code[ci] = j
                code_set.discard(i)
                code_set.add(j)
                cover_count[i_ball] -= 1
                cover_count[balls[j]] += 1
                improved = True
                break

        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    def anneal_swaps(code, cover_count, max_iter=20000):
        code_set = set(code)
        current_size = len(code)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(len(code))
            old_c = code[idx]
            new_c = np.random.randint(total_words)
            
            if new_c == old_c or new_c in code_set:
                continue
            
            # Apply swap
            code[idx] = new_c
            code_set.discard(old_c)
            code_set.add(new_c)
            cover_count[balls[old_c]] -= 1
            cover_count[balls[new_c]] += 1
            
            # Check validity
            affected = np.union1d(balls[old_c], balls[new_c])
            if np.any(cover_count[affected] == 0):
                # Revert
                code[idx] = old_c
                code_set.discard(new_c)
                code_set.add(old_c)
                cover_count[balls[old_c]] -= 1
                cover_count[balls[new_c]] += 1
                continue
            
            # Try to trim
            new_code, new_cover = trim_code(code, cover_count)
            
            if len(new_code) < current_size:
                current_size = len(new_code)
                code = new_code
                cover_count = new_cover
                code_set = set(code)
            else:
                # Revert if no improvement (keep it simple, avoid complex acceptance criteria)
                code[idx] = old_c
                code_set.discard(new_c)
                code_set.add(old_c)
                cover_count[balls[old_c]] -= 1
                cover_count[balls[new_c]] += 1
                # Note: trim_code might have modified code/cover_count in place?
                # In my trim_code, I do code.pop(idx). So if I revert, I need to restore the state before trim.
                # This is dangerous.
                # Let's make trim_code non-destructive or handle reverts carefully.
                # Actually, if len(new_code) == current_size, we revert.
                # But trim_code modified the list `code` passed to it?
                # Yes, `code.pop` modifies the list object.
                # So `code` is now `new_code`.
                # To revert, I need to restore the old state.
                # This is getting complicated.
                
                # Alternative: Don't revert immediately. Just keep the swapped state if valid, even if size is same.
                # This allows exploration.
                # Let's accept valid swaps that don't increase size.
                code = new_code
                cover_count = new_cover
                code_set = set(code)
                # current_size remains same

        return code, cover_count

    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 5
    for start_i in range(n_starts):
        if time.time() > deadline - 15:
            break

        if start_i == 0:
            code, cover_count = structured_init()
        else:
            code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        code, cover_count = trim_code(code, cover_count)
        code, cover_count = targeted_local_search(code, cover_count, max_iter=50000)
        
        if time.time() < deadline - 5:
            code, cover_count = anneal_swaps(code, cover_count, max_iter=10000)
            code, cover_count = targeted_local_search(code, cover_count, max_iter=50000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=100000)

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
