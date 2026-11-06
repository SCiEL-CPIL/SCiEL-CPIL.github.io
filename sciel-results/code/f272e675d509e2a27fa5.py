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

    # Precompute balls as numpy arrays for fast operations
    balls_np = np.zeros((N, ball_size), dtype=np.int32)
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
        balls_np[i] = neighbors

    balls = [list(balls_np[i]) for i in range(N)]

    # Precompute cover_map: for each word w, which codewords cover it
    cover_map = [list() for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered_mask = np.ones(N, dtype=np.bool_)
        uncovered_count = N
        in_code = np.zeros(N, dtype=np.bool_)
        code = []

        while uncovered_count > 0:
            # Compute marginal coverage for each candidate using numpy
            # marginal[c] = number of uncovered words in ball[c]
            marginals = np.sum(uncovered_mask[balls_np], axis=1)
            marginals[in_code] = -1

            best_c = int(np.argmax(marginals))
            if marginals[best_c] <= 0:
                break

            code.append(best_c)
            in_code[best_c] = True

            ball_c = balls_np[best_c]
            was_uncovered = uncovered_mask[ball_c]
            newly_covered = int(np.sum(was_uncovered))
            uncovered_mask[ball_c] = False
            uncovered_count -= newly_covered

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

    def hypergraph_core_peel(code):
        """
        Iteratively remove codewords whose removal doesn't leave any word
        in their ball with coverage count 1 (i.e., min coverage in ball >= 2).
        This is a 2-core peeling on the hypergraph of coverage.
        """
        if not code:
            return code

        cover_count = [0] * N
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        code_list = list(code)
        in_code = [False] * N
        for c in code_list:
            in_code[c] = True

        # Queue of codewords to check for removal
        # Start with all codewords; use a worklist
        from collections import deque
        worklist = deque(code_list)
        in_worklist = [False] * N
        for c in code_list:
            in_worklist[c] = True

        while worklist:
            c = worklist.popleft()
            in_worklist[c] = False
            if not in_code[c]:
                continue

            # Check if removing c leaves all words in ball[c] with coverage >= 2
            removable = True
            min_cov_after = ball_size + 1
            for w in balls[c]:
                cov_after = cover_count[w] - 1
                if cov_after < min_cov_after:
                    min_cov_after = cov_after
                if cov_after < 2:
                    removable = False
                    break

            if removable:
                # Remove c
                in_code[c] = False
                for w in balls[c]:
                    cover_count[w] -= 1

                # After removal, other codewords' balls may now have words with cov=1
                # that were previously cov=2. We need to re-check codewords that
                # share words with c's ball.
                # For each word w in ball[c], any codeword c2 covering w might now
                # have a word in its ball with cov=1. Add c2 to worklist.
                for w in balls[c]:
                    for c2 in cover_map[w]:
                        if in_code[c2] and not in_worklist[c2]:
                            worklist.append(c2)
                            in_worklist[c2] = True

        # Rebuild code_list from in_code
        result = [c for c in range(N) if in_code[c]]
        return result

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

                # Try 1-swap: find c_add not in code that covers all newly_uncovered
                found = -1
                if len(newly_uncovered) <= 4:
                    s = set(cover_map[newly_uncovered[0]])
                    for w in newly_uncovered[1:]:
                        s &= set(cover_map[w])
                        if not s:
                            break
                    for c_add in s:
                        if c_add != c_remove and not in_code[c_add]:
                            found = c_add
                            break

                if found >= 0:
                    code_list[i] = found
                    in_code[c_remove] = False
                    in_code[found] = True
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Restore cover counts
                    for w in balls[c_remove]:
                        cover_count[w] += 1

                    # Try 2-swap: remove c_remove and c_remove2, add c_add1 and c_add2
                    if time.time() < time_limit and len(code_list) >= 2:
                        # Find another codeword to remove that helps
                        for j in indices:
                            if j == i or time.time() > time_limit:
                                continue
                            c_remove2 = code_list[j]
                            # Temporarily remove both
                            nu2 = []
                            for w in balls[c_remove2]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0:
                                    nu2.append(w)
                            
                            all_uncovered = newly_uncovered + nu2
                            if len(all_uncovered) <= 5:
                                s = set(cover_map[all_uncovered[0]])
                                for w in all_uncovered[1:]:
                                    s &= set(cover_map[w])
                                    if not s:
                                        break
                                for c_add in s:
                                    if c_add not in (c_remove, c_remove2) and not in_code[c_add]:
                                        # Found a 2-swap
                                        code_list[i] = c_add
                                        code_list[j] = c_add  # placeholder, we need two adds
                                        # Actually for 2-swap we need to add 2 words
                                        # Let's just do: remove both, find one that covers all
                                        # This is a "2-remove, 1-add" which reduces size by 1
                                        code_list[i] = c_add
                                        code_list.pop(j)
                                        in_code[c_remove] = False
                                        in_code[c_remove2] = False
                                        in_code[c_add] = True
                                        # Restore cover counts properly
                                        for w in balls[c_add]:
                                            cover_count[w] += 1
                                        improved = True
                                        break
                            # Restore c_remove2's ball
                            for w in balls[c_remove2]:
                                cover_count[w] += 1
                            if improved:
                                break

        return code_list

    start_time = time.time()
    deadline = start_time + 115.0

    best_code = None
    best_len = float('inf')

    # Run multiple restarts with numpy-accelerated greedy
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        code = greedy_cover(i)
        # Apply hypergraph core peeling first (fast, no swaps)
        peeled = hypergraph_core_peel(code)
        refined = refine(peeled)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 5:
        ls_code = local_search(best_code, 42, deadline - 2)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code
        # Final core peel + refine
        final_peeled = hypergraph_core_peel(best_code)
        final_code = refine(final_peeled)
    else:
        final_code = best_code if best_code is not None else []

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
