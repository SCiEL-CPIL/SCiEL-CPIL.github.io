import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def is_valid(s_sorted, candidate):
        """Check if adding candidate to s_sorted keeps validity."""
        n = len(s_sorted)
        pos = bisect.bisect_left(s_sorted, candidate)

        # candidate as 'a': for all b,c in s_sorted with b > candidate, c >= b
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % candidate
            for k in range(i, n):
                if s_sorted[k] % candidate == target:
                    return False

        # candidate as 'b': for all a < candidate, c >= candidate
        for i in range(pos):
            a = s_sorted[i]
            if (2 * candidate) % a == 0:
                return False
            target = (-candidate) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    return False

        # candidate as 'c': for all a < b <= candidate
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > candidate:
                    break
                if (b + candidate) % a == 0:
                    return False

        return True

    def conflict_score(elem, s_sorted):
        """Count violations if elem were in the set."""
        n = len(s_sorted)
        pos = bisect.bisect_left(s_sorted, elem)
        count = 0

        # elem as 'a'
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % elem
            for k in range(i, n):
                if s_sorted[k] % elem == target:
                    count += 1

        # elem as 'b'
        for i in range(pos):
            a = s_sorted[i]
            if (2 * elem) % a == 0:
                count += 1
            target = (-elem) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    count += 1

        # elem as 'c'
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > elem:
                    break
                if (b + elem) % a == 0:
                    count += 1

        return count

    best_set = []

    # Phase 1: Try multiple starting points with repair
    for min_start in [128, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]:
        if time.time() - start_time > deadline - 20:
            break

        s_sorted = list(range(min_start, N + 1))
        s_set = set(s_sorted)

        # Repair: iteratively remove the element with highest conflict
        changed = True
        max_iter = 50
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1
            if len(s_sorted) <= 1:
                break

            max_conf = 0
            max_idx = -1
            for i, a in enumerate(s_sorted):
                conf = conflict_score(a, s_sorted)
                if conf > max_conf:
                    max_conf = conf
                    max_idx = i

            if max_idx >= 0 and max_conf > 0:
                removed = s_sorted.pop(max_idx)
                s_set.discard(removed)
                changed = True

        # Phase 2: Greedily add smaller elements
        for candidate in range(1, min_start):
            if time.time() - start_time > deadline - 15:
                break
            if candidate in s_set:
                continue
            if is_valid(s_sorted, candidate):
                pos = bisect.bisect_left(s_sorted, candidate)
                s_sorted.insert(pos, candidate)
                s_set.add(candidate)

        if len(s_sorted) > len(best_set):
            best_set = s_sorted[:]

    # Phase 3: Local search with swap operations
    s_set = set(best_set)
    for iteration in range(2000):
        if time.time() - start_time > deadline - 5:
            break

        if len(best_set) <= 1:
            break

        # Try a random swap: remove one element, try to add another
        s = best_set[:]
        s_local_set = set(s)
        idx = random.randint(0, len(s) - 1)
        removed = s.pop(idx)
        s_local_set.discard(removed)

        # Build candidate list: removed element + random others
        others = [x for x in range(1, N + 1) if x not in s_local_set]
        random.shuffle(others)
        candidates_to_try = [removed] + others[:30]
        random.shuffle(candidates_to_try)

        improved = False
        for candidate in candidates_to_try:
            if candidate in s_local_set:
                continue
            if is_valid(s, candidate):
                pos = bisect.bisect_left(s, candidate)
                s.insert(pos, candidate)
                s_local_set.add(candidate)
                if len(s) > len(best_set):
                    best_set = s[:]
                    s_set = s_local_set
                    improved = True
                    break

        # Also try pure additions without removal
        if not improved and random.random() < 0.3:
            s = best_set[:]
            s_local_set = set(s)
            others = [x for x in range(1, N + 1) if x not in s_local_set]
            random.shuffle(others)
            for candidate in others[:20]:
                if is_valid(s, candidate):
                    pos = bisect.bisect_left(s, candidate)
                    s.insert(pos, candidate)
                    s_local_set.add(candidate)
                    if len(s) > len(best_set):
                        best_set = s[:]
                        s_set = s_local_set
                        break

    return best_set
