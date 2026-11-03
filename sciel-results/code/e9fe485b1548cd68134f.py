import random
import time
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115

    total_words = q ** n

    # Precompute ball membership using integer representation
    # For q=3, n=7: each ball has 1 + n*(q-1) = 1+14 = 15 words

    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Build balls: balls[i] = list of words within Hamming distance 1 of word i
    balls = [[] for _ in range(total_words)]
    # Also build reverse: for each word w, which balls contain it (i.e., which codewords cover w)
    # This is the same as balls[w] since the relation is symmetric for radius 1
    # Actually balls[i] already gives us the neighbors. The reverse is the same set.

    all_words = [idx_to_tuple(i) for i in range(total_words)]

    for i in range(total_words):
        w = all_words[i]
        balls[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = tuple_to_idx(new_w)
                balls[i].append(j)

    ball_size = len(balls[0])  # 15 for q=3,n=7

    # Precompute for each word w, the list of codewords that cover it
    # coverers[w] = [c for c in range(total_words) if w in balls[c]]
    # This is the same as balls[w] because the relation is symmetric
    # But let's verify: w in balls[c] iff c in balls[w] for radius 1? Yes.
    # So coverers[w] = balls[w]

    def greedy_pq_run(seed):
        """Greedy set cover using a max-heap for O(log N) selection."""
        rng = random.Random(seed)
        uncovered = bytearray([1] * total_words)
        gains = [ball_size] * total_words
        selected = []
        remaining = total_words

        # Max-heap: store (-gain, tiebreaker, index)
        # We need to handle stale entries in the heap
        heap = [(-gains[i], rng.random(), i) for i in range(total_words)]
        heapq.heapify(heap)

        while remaining > 0:
            # Pop until we find a valid entry
            best_idx = -1
            best_gain = -1
            while heap:
                neg_g, _, idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[idx] and g > 0:
                    best_idx = idx
                    best_gain = g
                    break
                elif g == gains[idx]:
                    # gain is 0, skip
                    continue
                # else stale, skip

            if best_idx == -1:
                break

            selected.append(best_idx)

            # Update: for each word w in balls[best_idx] that is uncovered,
            # mark it covered and decrement gains for all codewords covering w
            for w in balls[best_idx]:
                if uncovered[w]:
                    uncovered[w] = 0
                    remaining -= 1
                    # For each codeword j that covers w, decrement gain
                    for j in balls[w]:
                        gains[j] -= 1
                        # Push updated gain into heap
                        if gains[j] > 0:
                            heapq.heappush(heap, (-gains[j], rng.random(), j))

        return selected

    def minimize_solution(solution):
        """Remove redundant codewords."""
        cover_count = [0] * total_words
        for c in solution:
            for w in balls[c]:
                cover_count[w] += 1

        current_sol = list(solution)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_sol) - 1, -1, -1):
                c = current_sol[i]
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    current_sol.pop(i)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
        return current_sol

    def swap_local_search(solution, max_iter=200, seed=0):
        """Try to reduce code size by swapping one codeword for another."""
        rng = random.Random(seed)
        sol = list(solution)
        sol_set = set(sol)

        # Build cover count
        cover_count = [0] * total_words
        for c in sol:
            for w in balls[c]:
                cover_count[w] += 1

        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1

            # For each codeword c in sol, find the words only c covers
            # Then try to find a replacement c' that covers all those words
            # and covers at least as many other words
            for i in range(len(sol)):
                c = sol[i]
                # Words uniquely covered by c
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)

                if not unique_words:
                    # c is redundant, just remove it
                    sol.pop(i)
                    sol_set.discard(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    improved = True
                    break

                # Find candidate replacements that cover all unique_words
                # and are not already in sol
                # A candidate c' must have all unique_words in balls[c']
                # This means c' must be within distance 1 of all unique_words
                # Since unique_words are all within distance 1 of c,
                # c' must be at distance <= 2 from c (triangle inequality)

                # For efficiency, just check candidates that cover all unique_words
                # Number of unique_words is small (usually 0-3)
                best_candidate = -1
                best_extra = -1

                # Candidate must cover all unique_words
                # Start with all words, filter
                for uw in unique_words:
                    # c' must be in balls[uw]
                    # Intersect with current candidate set
                    if best_candidate == -1:
                        candidates = [j for j in balls[uw] if j not in sol_set]
                    else:
                        # Keep only those also in balls[uw]
                        uw_set = set(balls[uw])
                        candidates = [j for j in candidates if j in uw_set]

                    if not candidates:
                        break

                if candidates:
                    # Evaluate each candidate: how many words does it newly cover?
                    # Words with cover_count 0 after removing c
                    # Actually, after removing c, words in balls[c] with cover_count 1 become uncovered
                    # So we need c' to cover those, and ideally cover more

                    # Count extra coverage: words in balls[c'] with cover_count >= 1
                    # that are not in unique_words (since those are mandatory)
                    uw_set = set(unique_words)
                    for c_prime in candidates:
                        extra = 0
                        for w in balls[c_prime]:
                            if w not in uw_set and cover_count[w] >= 1:
                                extra += 1
                        if extra > best_extra:
                            best_extra = extra
                            best_candidate = c_prime

                    if best_candidate != -1:
                        # Perform swap
                        sol[i] = best_candidate
                        sol_set.discard(c)
                        sol_set.add(best_candidate)
                        # Update cover counts
                        for w in balls[c]:
                            cover_count[w] -= 1
                        for w in balls[best_candidate]:
                            cover_count[w] += 1
                        improved = True
                        break

            # Check time
            if time.time() > deadline - 5:
                break

        return sol

    best_solution = None
    best_len = float('inf')

    # Phase 1: Run greedy with PQ, multiple seeds
    num_greedy = 30
    for s in range(num_greedy):
        if time.time() > deadline - 30:
            break
        sol = greedy_pq_run(s)
        sol = minimize_solution(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search (swap) on best solutions
    if best_solution is not None and time.time() < deadline - 10:
        # Run swap local search on the best solution
        sol2 = swap_local_search(best_solution, max_iter=500, seed=42)
        sol2 = minimize_solution(sol2)
        if len(sol2) < best_len:
            best_len = len(sol2)
            best_solution = sol2

    # Phase 3: Try swap on a few more greedy results
    if time.time() < deadline - 10:
        for s in [100, 200, 300]:
            if time.time() > deadline - 10:
                break
            sol = greedy_pq_run(s)
            sol = minimize_solution(sol)
            sol = swap_local_search(sol, max_iter=300, seed=s)
            sol = minimize_solution(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    if best_solution is None:
        return [idx_to_tuple(i) for i in range(total_words)]

    return [idx_to_tuple(i) for i in best_solution]
