import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

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

    # Precompute balls
    balls = []
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

    # Precompute cover_map: for each word w, which codewords cover it
    cover_map = [list() for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # Convert balls to tuples for faster hashing if needed, or keep as lists
    # For local search, we need to check membership in balls[c_add].
    # Precomputing sets is memory heavy (2187 sets of 15 ints). 
    # 2187 * 15 * 8 bytes (approx) = 262KB. This is fine.
    ball_sets = [set(b) for b in balls]

    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N
        code = []
        uncovered_count = N
        
        # Max-heap: (-cnt[i], i)
        # We need to handle updates. We'll push new values and ignore stale ones.
        heap = [(-cnt[i], i) for i in range(N)]
        heapq.heapify(heap)
        
        in_code = [False] * N

        while uncovered_count > 0:
            # Pop valid candidate
            while heap:
                neg_val, i = heapq.heappop(heap)
                # Check if this entry is still valid (cnt[i] matches and not in code)
                if cnt[i] == -neg_val and not in_code[i]:
                    break
            else:
                break # Should not happen if uncovered > 0

            c = i
            code.append(c)
            in_code[c] = True

            ball_c = balls[c]
            for w in ball_c:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1
                        if cnt[c2] > 0 and not in_code[c2]:
                            heapq.heappush(heap, (-cnt[c2], c2))

        return code

    def refine(code):
        if not code:
            return code

        cover_count = [0] * N
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
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = [0] * N
        for c in code_list:
            for w in balls[c]:
                cover_count[w] += 1

        in_code = [False] * N
        for c in code_list:
            in_code[c] = True

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
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
                    break

                # Try 1-swap
                # Find a codeword c_add not in code that covers all newly_uncovered
                found = -1
                # Heuristic: check candidates that cover the most rare uncovered words
                # Or just iterate all N? N=2187. 
                # 2187 checks * 15 membership checks = ~33k ops. 
                # This is fast enough in Python if done in C-optimized loops? No, pure Python.
                # 33k ops is ~0.001s. 
                # 186 codewords * 0.001s = 0.18s per pass.
                # We have 120s. So ~600 passes. This is good.
                
                for c_add in range(N):
                    if c_add == c_remove or in_code[c_add]:
                        continue
                    # Check if ball_sets[c_add] covers all newly_uncovered
                    # Optimization: check if len(intersection) == len(newly_uncovered)
                    # Or just: all(w in ball_sets[c_add] for w in newly_uncovered)
                    if all(w in ball_sets[c_add] for w in newly_uncovered):
                        found = c_add
                        break
                
                if found >= 0:
                    # Perform swap
                    code_list[i] = found
                    in_code[c_remove] = False
                    in_code[found] = True
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Revert deletion
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_code = None
    best_len = float('inf')

    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        code = greedy_cover(i)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 5:
        # Local search on the best code
        ls_code = local_search(best_code, 42, deadline - 2)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

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
