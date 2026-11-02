import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute digits
    digits_table = np.zeros((N, 7), dtype=np.int8)
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits_table[i, pos] = tmp % 3
            tmp //= 3

    # Precompute ball indices: for each word, list of words within Hamming distance 1
    # ball_indices[i] = array of indices within distance 1 of i
    ball_indices = [None] * N
    for i in range(N):
        neighbors = [i]
        for pos in range(7):
            orig = digits_table[i, pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    neighbors.append(new_i)
        ball_indices[i] = np.array(neighbors, dtype=np.int32)

    # Precompute reverse: for each word w, which codewords cover it
    # coverers[w] = list of codeword indices c such that w is in ball_indices[c]
    # This is the same as ball_indices due to symmetry of Hamming distance
    # So coverers[w] = ball_indices[w]

    full_mask = (1 << N) - 1

    def greedy_init(seed):
        """Greedy set cover with bitmasks."""
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        # Precompute ball_masks for greedy
        ball_masks = [0] * N
        for i in range(N):
            mask = 0
            for w in ball_indices[i]:
                mask |= (1 << w)
            ball_masks[i] = mask

        while uncovered:
            best_idx = -1
            best_count = -1
            candidates = list(range(N))
            rng.shuffle(candidates)

            for i in candidates:
                if i in selected_set:
                    continue
                cnt = (uncovered & ball_masks[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    if best_count == uncovered.bit_count():
                        break

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]

            if time.time() > deadline - 15.0:
                break

        return selected

    def structured_init():
        """
        Use tetracode structure: the optimal ternary code of length 4 with radius 1
        has 9 words. For length 7, we can use a product-like construction.
        Actually, let's use a different structured approach:
        Start with all 27 words of the form (a,b,c,d,e,f,g) where (a,b,c,d) is a tetracode word
        and (e,f,g) varies. That gives 9*27 = 243 words.
        Then prune heavily.
        
        Better: use a greedy approach but seeded with structured candidates.
        """
        # Tetracode: 9 words of length 4
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])

        # Build initial code: for each tetracode word, add all 27 extensions
        # This gives 9 * 27 = 243 codewords
        code = []
        for t in tetracode:
            for e in range(3):
                for f in range(3):
                    for g in range(3):
                        word = t + [e, f, g]
                        idx = 0
                        for pos, d in enumerate(word):
                            idx += d * p3[pos]
                        code.append(idx)

        return code

    def compute_coverage(code):
        """Compute coverage count array using numpy."""
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in ball_indices[c]:
                cnt[w] += 1
        return cnt

    def is_valid(cnt):
        """Check if all words are covered."""
        return np.all(cnt >= 1)

    def simulated_annealing(init_code, seed, time_budget):
        """
        Simulated annealing on the codeword set.
        Moves:
        1. Delete a codeword (if valid)
        2. Swap: remove one codeword, add another (if valid)
        3. 2-for-1: remove two, add one (if valid)
        4. 1-for-2: remove one, add two (if valid, for diversification)
        """
        rng = random.Random(seed)
        code = list(init_code)
        code_set = set(code)
        cnt = compute_coverage(code)

        best_code = code[:]
        best_size = len(code)

        # SA parameters
        T = 5.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        max_iterations = 100000

        iter_count = 0
        while time.time() < time_budget and iter_count < max_iterations:
            iter_count += 1
            current_size = len(code)
            improved = False

            # Try multiple moves per iteration
            for _ in range(10):
                if time.time() > time_budget:
                    break

                move_type = rng.random()
                new_code = None
                new_cnt = None

                if move_type < 0.3 and current_size > 0:
                    # Deletion
                    c = rng.choice(code)
                    # Check if c is essential
                    essential = False
                    for w in ball_indices[c]:
                        if cnt[w] == 1:
                            essential = True
                            break
                    if not essential:
                        # Remove c
                        new_code = [x for x in code if x != c]
                        new_cnt = cnt.copy()
                        for w in ball_indices[c]:
                            new_cnt[w] -= 1
                        if np.all(new_cnt >= 1):
                            code = new_code
                            code_set.discard(c)
                            cnt = new_cnt
                            improved = True
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                            break
                        else:
                            # Accept with probability
                            if rng.random() < np.exp(-1.0 / max(T, 0.01)):
                                code = new_code
                                code_set.discard(c)
                                cnt = new_cnt

                elif move_type < 0.7 and current_size > 0:
                    # 1-for-1 Swap
                    c = rng.choice(code)
                    # Find words uniquely covered by c
                    unique_words = [w for w in ball_indices[c] if cnt[w] == 1]

                    if not unique_words:
                        # c is redundant, just remove it
                        new_code = [x for x in code if x != c]
                        new_cnt = cnt.copy()
                        for w in ball_indices[c]:
                            new_cnt[w] -= 1
                        if np.all(new_cnt >= 1):
                            code = new_code
                            code_set.discard(c)
                            cnt = new_cnt
                            improved = True
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                            break
                        continue

                    # Find candidates that cover all unique_words
                    # Candidate c' must be within distance 1 of each word in unique_words
                    # i.e., c' must be in the intersection of balls of unique_words
                    # c' covers w iff w is in ball_indices[c'] iff c' is in ball_indices[w]
                    # So c' must be in intersection of ball_indices[w] for w in unique_words

                    # Start with ball of first unique word
                    cand_set = set(ball_indices[unique_words[0]])
                    for w in unique_words[1:]:
                        cand_set &= set(ball_indices[w])
                        if not cand_set:
                            break

                    if not cand_set:
                        continue

                    # Remove c from candidates
                    cand_set.discard(c)
                    # Remove existing codewords
                    cand_set -= code_set

                    if not cand_set:
                        continue

                    c_prime = rng.choice(list(cand_set))

                    # Perform swap
                    new_code = [x for x in code if x != c] + [c_prime]
                    new_cnt = cnt.copy()
                    for w in ball_indices[c]:
                        new_cnt[w] -= 1
                    for w in ball_indices[c_prime]:
                        new_cnt[w] += 1

                    if np.all(new_cnt >= 1):
                        code = new_code
                        code_set.discard(c)
                        code_set.add(c_prime)
                        cnt = new_cnt
                        improved = True
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        break
                    else:
                        # Accept with probability based on number of uncovered words
                        uncovered_count = np.sum(new_cnt < 1)
                        if uncovered_count <= 3 and rng.random() < np.exp(-uncovered_count / max(T, 0.01)):
                            code = new_code
                            code_set.discard(c)
                            code_set.add(c_prime)
                            cnt = new_cnt

                elif move_type < 0.85 and current_size >= 2:
                    # 2-for-1: remove two, add one
                    i1, i2 = rng.sample(range(len(code)), 2)
                    c1, c2 = code[i1], code[i2]

                    # Words uniquely covered by c1 or c2
                    unique_to_pair = set()
                    for w in ball_indices[c1]:
                        if cnt[w] == 1:
                            unique_to_pair.add(w)
                    for w in ball_indices[c2]:
                        if cnt[w] == 1:
                            unique_to_pair.add(w)

                    if not unique_to_pair:
                        # Both redundant, remove one
                        new_code = [x for x in code if x != c2]
                        new_cnt = cnt.copy()
                        for w in ball_indices[c2]:
                            new_cnt[w] -= 1
                        if np.all(new_cnt >= 1):
                            code = new_code
                            code_set.discard(c2)
                            cnt = new_cnt
                            improved = True
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                            break
                        continue

                    # Find c3 that covers all unique_to_pair
                    unique_list = list(unique_to_pair)
                    cand_set = set(ball_indices[unique_list[0]])
                    for w in unique_list[1:]:
                        cand_set &= set(ball_indices[w])
                        if not cand_set:
                            break

                    if not cand_set:
                        continue

                    cand_set.discard(c1)
                    cand_set.discard(c2)
                    cand_set -= code_set

                    if not cand_set:
                        continue

                    c3 = rng.choice(list(cand_set))

                    new_code = [x for x in code if x != c1 and x != c2] + [c3]
                    new_cnt = cnt.copy()
                    for w in ball_indices[c1]:
                        new_cnt[w] -= 1
                    for w in ball_indices[c2]:
                        new_cnt[w] -= 1
                    for w in ball_indices[c3]:
                        new_cnt[w] += 1

                    if np.all(new_cnt >= 1):
                        code = new_code
                        code_set.discard(c1)
                        code_set.discard(c2)
                        code_set.add(c3)
                        cnt = new_cnt
                        improved = True
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        break
                    else:
                        uncovered_count = np.sum(new_cnt < 1)
                        if uncovered_count <= 5 and rng.random() < np.exp(-uncovered_count / max(T, 0.01)):
                            code = new_code
                            code_set.discard(c1)
                            code_set.discard(c2)
                            code_set.add(c3)
                            cnt = new_cnt

                else:
                    # 1-for-2: remove one, add two (diversification)
                    if current_size == 0:
                        continue
                    c = rng.choice(code)
                    # Find two random candidates not in code
                    all_candidates = [i for i in range(N) if i not in code_set]
                    if len(all_candidates) < 2:
                        continue
                    c1, c2 = rng.sample(all_candidates, 2)

                    new_code = [x for x in code if x != c] + [c1, c2]
                    new_cnt = cnt.copy()
                    for w in ball_indices[c]:
                        new_cnt[w] -= 1
                    for w in ball_indices[c1]:
                        new_cnt[w] += 1
                    for w in ball_indices[c2]:
                        new_cnt[w] += 1

                    if np.all(new_cnt >= 1):
                        code = new_code
                        code_set.discard(c)
                        code_set.add(c1)
                        code_set.add(c2)
                        cnt = new_cnt
                        # This increases size, so don't update best
                        break

            # Cool down
            T = max(T * cooling, T_min)

            # If no improvement for a while, perturb
            if not improved and iter_count % 100 == 0:
                # Random perturbation: swap 3 random codewords
                if len(code) >= 3:
                    for _ in range(3):
                        c = rng.choice(code)
                        all_candidates = [i for i in range(N) if i not in code_set]
                        if not all_candidates:
                            break
                        c_new = rng.choice(all_candidates)
                        code.remove(c)
                        code_set.discard(c)
                        code.append(c_new)
                        code_set.add(c_new)
                        for w in ball_indices[c]:
                            cnt[w] -= 1
                        for w in ball_indices[c_new]:
                            cnt[w] += 1

        return best_code

    def local_search_prune(code):
        """Aggressive local search: repeated deletion and swap until no improvement."""
        code = list(code)
        code_set = set(code)
        cnt = compute_coverage(code)
        best_code = code[:]
        best_size = len(code)

        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            improved = False

            # Deletion phase
            to_remove = []
            for c in code:
                is_essential = False
                for w in ball_indices[c]:
                    if cnt[w] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.discard(c)
                    for w in ball_indices[c]:
                        cnt[w] -= 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                improved = True
                continue

            # Swap phase
            swapped = False
            for c in list(code):
                unique_words = [w for w in ball_indices[c] if cnt[w] == 1]
                if not unique_words:
                    code.remove(c)
                    code_set.discard(c)
                    for w in ball_indices[c]:
                        cnt[w] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    swapped = True
                    improved = True
                    break

                cand_set = set(ball_indices[unique_words[0]])
                for w in unique_words[1:]:
                    cand_set &= set(ball_indices[w])
                    if not cand_set:
                        break
                cand_set.discard(c)
                cand_set -= code_set

                if cand_set:
                    c_prime = next(iter(cand_set))  # Take first candidate
                    # Verify
                    new_cnt = cnt.copy()
                    for w in ball_indices[c]:
                        new_cnt[w] -= 1
                    for w in ball_indices[c_prime]:
                        new_cnt[w] += 1
                    if np.all(new_cnt >= 1):
                        code.remove(c)
                        code_set.discard(c)
                        code.append(c_prime)
                        code_set.add(c_prime)
                        cnt = new_cnt
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        swapped = True
                        improved = True
                        break

            if not improved:
                break

        return best_code

    # Main search
    best_overall = None
    best_size = float('inf')

    # First: structured init + local search
    struct_code = structured_init()
    pruned = local_search_prune(struct_code)
    if len(pruned) < best_size:
        best_size = len(pruned)
        best_overall = pruned[:]

    # Then: SA from structured init
    sa_budget = min(deadline - 10.0, time.time() + 30.0)
    sa_result = simulated_annealing(struct_code, 123, sa_budget)
    if len(sa_result) < best_size:
        best_size = len(sa_result)
        best_overall = sa_result[:]

    # Then: multiple greedy inits + SA
    seed = 42
    while time.time() < deadline - 15.0:
        init_code = greedy_init(seed)
        seed += 1
        if not init_code:
            break

        # Quick local search
        pruned = local_search_prune(init_code)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_overall = pruned[:]

        # SA with remaining time
        sa_budget = min(deadline - 10.0, time.time() + 20.0)
        sa_result = simulated_annealing(pruned, seed, sa_budget)
        seed += 1
        if len(sa_result) < best_size:
            best_size = len(sa_result)
            best_overall = sa_result[:]

        if best_size <= 170:
            break

    # Final local search on best
    if best_overall:
        final = local_search_prune(best_overall)
        if len(final) < best_size:
            best_size = len(final)
            best_overall = final[:]

    if not best_overall:
        # Fallback: pure greedy
        init_code = greedy_init(0)
        best_overall = local_search_prune(init_code)

    # Convert back to lists of integers (most-significant digit first)
    result = []
    for idx in best_overall:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        digits.reverse()
        result.append(digits)

    return result


def _construct_general(q, n, radius, deadline):
    """Simple greedy + prune for general parameters."""
    total = q ** n
    words = list(range(total))

    def decode(val):
        return [val // (q ** (n - i - 1)) % q for i in range(n)]

    def encode(digits):
        val = 0
        for i, d in enumerate(digits):
            val += d * (q ** (n - i - 1))
        return val

    balls = {}
    for w in words:
        digs = decode(w)
        neighbors = {w}
        for pos in range(n):
            original = digs[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digs = digs[:]
                    new_digs[pos] = new_val
                    neighbors.add(encode(new_digs))
        balls[w] = neighbors

    uncovered = set(words)
    selected = []
    while uncovered and time.time() < deadline - 5.0:
        best_w = max(words, key=lambda w: len(balls[w] & uncovered))
        selected.append(best_w)
        uncovered -= balls[best_w]

    if selected:
        counts = {w: 0 for w in words}
        for c in selected:
            for w in balls[c]:
                counts[w] += 1

        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                can_remove = all(counts[w] > 1 for w in balls[c])
                if can_remove:
                    for w in balls[c]:
                        counts[w] -= 1
                    selected.pop(i)
                    changed = True

    return [decode(w) for w in selected]

# EVOLVE-BLOCK-END
