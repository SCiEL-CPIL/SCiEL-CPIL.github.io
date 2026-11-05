import numpy as np
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    N = 3 ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    # Precompute conversion tables
    def int_to_base3(num, length=7):
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

    # Precompute balls using numpy for speed
    all_digits = np.array([int_to_base3(i, n) for i in range(N)])
    
    # For each word, compute its ball (itself + all single-position changes)
    balls = []
    for i in range(N):
        d = all_digits[i]
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for nv in range(q):
                if nv != orig:
                    nd = d.copy()
                    nd[pos] = nv
                    neighbors.append(base3_to_int(nd))
        balls.append(neighbors)
    
    # Convert balls to sets for faster intersection checks
    ball_sets = [set(b) for b in balls]

    # Precompute cover_map: for each word w, list of codewords c whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    def greedy_cover(seed):
        """Greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N
        code = []
        uncovered_count = N

        # Use a bucket approach for max-finding
        # cnt values range from 0 to ball_size (15)
        buckets = [[] for _ in range(ball_size + 1)]
        for i in range(N):
            buckets[cnt[i]].append(i)
        max_cnt = ball_size

        while uncovered_count > 0:
            # Find max cnt
            while max_cnt > 0 and not buckets[max_cnt]:
                max_cnt -= 1
            if max_cnt == 0:
                break
            candidates = buckets[max_cnt]
            c = rng.choice(candidates)
            code.append(c)
            buckets[max_cnt].remove(c)

            ball_c = balls[c]
            for w in ball_c:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        if cnt[c2] > 0:
                            buckets[cnt[c2]].remove(c2)
                            cnt[c2] -= 1
                            buckets[cnt[c2]].append(c2)

            if max_cnt > 0:
                # max_cnt might have changed
                while max_cnt > 0 and not buckets[max_cnt]:
                    max_cnt -= 1

        return code

    def refine(code):
        """Remove redundant codewords."""
        if not code:
            return code

        cover_count = [0] * N
        code_set = set(code)
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        code_list = list(code)
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

    def local_search(code, seed, time_limit):
        """1-swap local search: try replacing one codeword with another."""
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = [0] * N
        code_set = set(code_list)
        for c in code_list:
            for w in balls[c]:
                cover_count[w] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                
                # Temporarily remove
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                
                # Check if still covered
                uncovered = []
                for w in range(N):
                    if cover_count[w] == 0:
                        uncovered.append(w)
                
                if not uncovered:
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    improved = True
                    break

                # Try to find a replacement
                found = -1
                for c_add in range(N):
                    if c_add == c_remove or c_add in code_set:
                        continue
                    # Check if c_add covers all uncovered
                    ball_add = ball_sets[c_add]
                    if all(w in ball_add for w in uncovered):
                        found = c_add
                        break
                
                if found >= 0:
                    code_list[i] = found
                    code_set.add(found)
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Restore
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    def two_swap_search(code, seed, time_limit):
        """Try removing two codewords and adding one or two to cover the gap."""
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = [0] * N
        code_set = set(code_list)
        for c in code_list:
            for w in balls[c]:
                cover_count[w] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            if len(code_list) < 2:
                break
            i, j = rng.sample(range(len(code_list)), 2)
            c1 = code_list[i]
            c2 = code_list[j]

            # Remove both
            for w in balls[c1]:
                cover_count[w] -= 1
            for w in balls[c2]:
                cover_count[w] -= 1

            uncovered = [w for w in range(N) if cover_count[w] == 0]
            
            if not uncovered:
                # Remove both
                code_list.pop(max(i, j))
                code_list.pop(min(i, j))
                code_set.discard(c1)
                code_set.discard(c2)
                improved = True
                continue

            # Try adding one codeword that covers all uncovered
            found = -1
            for c_add in range(N):
                if c_add in code_set:
                    continue
                ball_add = ball_sets[c_add]
                if all(w in ball_add for w in uncovered):
                    found = c_add
                    break
            
            if found >= 0:
                # Replace c1 with found, remove c2
                code_list[i] = found
                code_list.pop(j if j > i else i)
                code_set.discard(c2)
                code_set.add(found)
                for w in balls[found]:
                    cover_count[w] += 1
                improved = True
                continue

            # Try adding two codewords
            found1 = -1
            found2 = -1
            for c_add1 in range(N):
                if c_add1 in code_set:
                    continue
                ball_add1 = ball_sets[c_add1]
                remaining_uncovered = [w for w in uncovered if w not in ball_add1]
                if not remaining_uncovered:
                    found1 = c_add1
                    found2 = -1
                    break
                for c_add2 in range(N):
                    if c_add2 in code_set or c_add2 == c_add1:
                        continue
                    ball_add2 = ball_sets[c_add2]
                    if all(w in ball_add2 for w in remaining_uncovered):
                        found1 = c_add1
                        found2 = c_add2
                        break
                if found1 >= 0:
                    break
            
            if found1 >= 0:
                if found2 >= 0:
                    code_list[i] = found1
                    code_list[j] = found2
                    code_set.add(found1)
                    code_set.add(found2)
                    for w in balls[found1]:
                        cover_count[w] += 1
                    for w in balls[found2]:
                        cover_count[w] += 1
                else:
                    code_list[i] = found1
                    code_list.pop(j if j > i else i)
                    code_set.add(found1)
                    for w in balls[found1]:
                        cover_count[w] += 1
                improved = True
            else:
                # Restore
                for w in balls[c1]:
                    cover_count[w] += 1
                for w in balls[c2]:
                    cover_count[w] += 1

        return code_list

    # Main search
    best_code = None
    best_len = float('inf')

    # Phase 1: Multiple greedy restarts
    num_restarts = 60
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        code = greedy_cover(i)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is None:
        return []

    # Phase 2: Local search on best
    if time.time() < deadline - 15:
        ls_code = local_search(best_code, 42, deadline - 10)
        ls_code = refine(ls_code)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    # Phase 3: Two-swap search
    if time.time() < deadline - 5:
        ts_code = two_swap_search(best_code, 123, deadline - 3)
        ts_code = refine(ts_code)
        if len(ts_code) < best_len:
            best_len = len(ts_code)
            best_code = ts_code

    # Final refinement
    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []

    result = []
    for c in final_code:
        result.append(int_to_base3(c, n))

    return result

def _generic_greedy(q, n, radius):
    N = q ** n
    
    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    result = []
    for c in code:
        digits = int_to_base(c, n, q)
        result.append(digits)
    return result
