import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2 * n  # 17
    powers = [3**i for i in range(n)]

    # Precompute balls as a list of lists of indices
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            # +1
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            # -1
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball

    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    # Greedy Set Cover
    def greedy_set_cover(initial_code=None):
        code = set(initial_code) if initial_code else set()
        uncovered = np.ones(M, dtype=bool)
        gains = np.full(M, ball_size, dtype=np.int32)

        # Initialize uncovered and gains based on initial code
        if initial_code:
            for c in initial_code:
                for p in balls[c]:
                    if uncovered[p]:
                        uncovered[p] = False
                        for c_prime in coverers[p]:
                            if c_prime not in code:
                                gains[c_prime] -= 1

        # Heap entries: (-gain, codeword)
        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
        heapq.heapify(heap)

        uncovered_count = int(np.sum(uncovered))

        while uncovered_count > 0:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if c in code:
                    continue
                if -neg_g != gains[c]:
                    continue
                if gains[c] <= 0:
                    continue
                valid_c = c
                break

            if valid_c is None:
                break

            code.add(valid_c)

            # Update uncovered and gains
            for p in balls[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    # Decrease gain for all codewords covering p
                    for c_prime in coverers[p]:
                        if c_prime not in code:
                            gains[c_prime] -= 1
                            if gains[c_prime] > 0:
                                heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code, uncovered, gains

    # Local Search
    def local_search(code, time_limit=115):
        start_time = time.time()

        # Precompute count
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1

        # Helper to check if c is redundant
        def is_redundant(c):
            for p in balls[c]:
                if count[p] == 1:
                    return False
            return True

        # Helper to remove c
        def remove_c(c):
            code.remove(c)
            for p in balls[c]:
                count[p] -= 1

        # Helper to add c
        def add_c(c):
            code.add(c)
            for p in balls[c]:
                count[p] += 1

        improved = True

        while improved and (time.time() - start_time) < time_limit:
            improved = False

            # Phase 1: Remove redundant
            to_remove = []
            for c in list(code):
                if is_redundant(c):
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    remove_c(c)
                improved = True
                continue

            # Phase 2: 1-Swap
            code_list = list(code)
            random.shuffle(code_list)

            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break

                unique_points = [p for p in balls[c] if count[p] == 1]

                if not unique_points:
                    remove_c(c)
                    improved = True
                    swapped = True
                    break

                # Find candidates for c'
                candidates = set(coverers[unique_points[0]])
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers[p])

                candidates -= code

                if candidates:
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    c_new = cand_list[0]

                    remove_c(c)
                    add_c(c_new)
                    improved = True
                    swapped = True
                    break

            if swapped:
                continue

            # Phase 3: 2-Swap
            if (time.time() - start_time) < time_limit:
                code_list = list(code)
                if len(code_list) >= 2:
                    c1, c2 = random.sample(code_list, 2)

                    U1 = [p for p in balls[c1] if count[p] == 1]
                    U2 = [p for p in balls[c2] if count[p] == 1]

                    # Case 1: One new codeword covers both U1 and U2
                    if U1 and U2:
                        all_unique = set(U1) | set(U2)
                        first_p = next(iter(all_unique))
                        cands = set(coverers[first_p])
                        for p in all_unique:
                            cands.intersection_update(coverers[p])

                        cands -= code
                        cands.discard(c1)
                        cands.discard(c2)

                        if cands:
                            c_new = random.choice(list(cands))
                            remove_c(c1)
                            remove_c(c2)
                            add_c(c_new)
                            improved = True
                            continue

                    # Case 2: Two new codewords
                    cands1 = set(coverers[U1[0]]) if U1 else set()
                    if U1:
                        for p in U1[1:]:
                            cands1.intersection_update(coverers[p])
                    cands1 -= code
                    cands1.discard(c1)
                    cands1.discard(c2)

                    cands2 = set(coverers[U2[0]]) if U2 else set()
                    if U2:
                        for p in U2[1:]:
                            cands2.intersection_update(coverers[p])
                    cands2 -= code
                    cands2.discard(c1)
                    cands2.discard(c2)

                    if cands1 and cands2:
                        c1_new = random.choice(list(cands1))
                        valid_c2s = [c for c in cands2 if c != c1_new]
                        if valid_c2s:
                            c2_new = random.choice(valid_c2s)

                            remove_c(c1)
                            remove_c(c2)
                            add_c(c1_new)
                            add_c(c2_new)
                            improved = True
                            continue

            # Phase 4: 3-Swap (Simple version)
            if (time.time() - start_time) < time_limit:
                code_list = list(code)
                if len(code_list) >= 3:
                    c1, c2, c3 = random.sample(code_list, 3)
                    
                    U1 = [p for p in balls[c1] if count[p] == 1]
                    U2 = [p for p in balls[c2] if count[p] == 1]
                    U3 = [p for p in balls[c3] if count[p] == 1]
                    
                    # Try to replace with 3 new codewords
                    # This is heuristic: find one for each
                    cands1 = set(coverers[U1[0]]) if U1 else set()
                    if U1:
                        for p in U1[1:]:
                            cands1.intersection_update(coverers[p])
                    cands1 -= code
                    cands1.discard(c1); cands1.discard(c2); cands1.discard(c3)
                    
                    cands2 = set(coverers[U2[0]]) if U2 else set()
                    if U2:
                        for p in U2[1:]:
                            cands2.intersection_update(coverers[p])
                    cands2 -= code
                    cands2.discard(c1); cands2.discard(c2); cands2.discard(c3)
                    
                    cands3 = set(coverers[U3[0]]) if U3 else set()
                    if U3:
                        for p in U3[1:]:
                            cands3.intersection_update(coverers[p])
                    cands3 -= code
                    cands3.discard(c1); cands3.discard(c2); cands3.discard(c3)
                    
                    if cands1 and cands2 and cands3:
                        # Pick distinct candidates
                        c1_new = random.choice(list(cands1))
                        cands2.discard(c1_new)
                        cands3.discard(c1_new)
                        if cands2 and cands3:
                            c2_new = random.choice(list(cands2))
                            cands3.discard(c2_new)
                            if cands3:
                                c3_new = random.choice(list(cands3))
                                remove_c(c1)
                                remove_c(c2)
                                remove_c(c3)
                                add_c(c1_new)
                                add_c(c2_new)
                                add_c(c3_new)
                                improved = True
                                continue

            # Perturbation
            if (time.time() - start_time) < time_limit:
                k = 5
                if len(code) > k:
                    to_perturb = random.sample(list(code), k)
                    for c in to_perturb:
                        remove_c(c)

                    # Re-add greedily
                    # Recalculate gains for remaining code
                    import heapq
                    gains = np.zeros(M, dtype=np.int32)
                    for c in range(M):
                        if c not in code:
                            g = 0
                            for p in balls[c]:
                                if count[p] == 0:
                                    g += 1
                            gains[c] = g

                    heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
                    heapq.heapify(heap)

                    while heap:
                        neg_g, c = heapq.heappop(heap)
                        if c in code:
                            continue
                        if -neg_g != gains[c]:
                            continue
                        if gains[c] <= 0:
                            continue

                        add_c(c)

                        for p in balls[c]:
                            if count[p] == 1:  # Just became covered
                                for c_prime in coverers[p]:
                                    if c_prime not in code:
                                        gains[c_prime] -= 1
                                        if gains[c_prime] > 0:
                                            heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code

    # Multi-start local search with symmetry fixing
    best_code = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 115

    # Try with symmetry fixing (force codeword 0)
    initial_code = {0}
    code, uncovered, gains = greedy_set_cover(initial_code)
    code = local_search(code, time_limit=10)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # Try multiple random starts
    for _ in range(5):
        if time.time() - start_time > time_limit - 10:
            break

        # Random initial code
        initial_code = set(random.sample(range(M), 10))
        code, uncovered, gains = greedy_set_cover(initial_code)
        code = local_search(code, time_limit=10)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()

    # Final refinement on best code
    if best_code:
        best_code = local_search(best_code, time_limit=time_limit - (time.time() - start_time))

    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
