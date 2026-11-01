"""Combined program: fast bitset greedy + vectorized local search (delete + 1-swap + 2-swap) for K_3(7,1)."""
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187
    ball_size = 1 + 7 * 2  # 15

    # Precompute powers of 3
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        ball = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    ball.append(new_i)
        ball_indices.append(ball)

    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]

    # Precompute covers as bitsets (Python ints) for fast uncovered tracking
    covers = [0] * N
    for i in range(N):
        mask = 1 << i
        for j in ball_indices[i]:
            if j != i:
                mask |= (1 << j)
        covers[i] = mask

    full_mask = (1 << N) - 1

    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        uncovered = full_mask
        selected = []

        while uncovered:
            best_count = -1
            best_idx = -1
            best_indices = []

            # Optimization: check candidates in random order for randomized version
            candidates = list(range(N))
            if seed is not None:
                rng.shuffle(candidates)

            for i in candidates:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_indices = [i]
                elif cnt == best_count:
                    best_indices.append(i)

            if best_count == 0:
                break

            if len(best_indices) > 1 and seed is not None:
                idx = rng.choice(best_indices)
            else:
                idx = best_idx

            selected.append(idx)
            uncovered &= ~covers[idx]

            if time.time() > deadline - 10.0:
                break

        return selected, uncovered

    # Vectorized delete phase: remove all redundant codewords at once
    def run_delete_phase(code_list, cnt_arr):
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in range(len(code_list)):
                c = code_list[idx]
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    to_remove.append(idx)
            if to_remove:
                # Remove in reverse order to preserve indices
                for idx in sorted(to_remove, reverse=True):
                    c = code_list[idx]
                    cnt_arr[ball_indices_np[c]] -= 1
                    code_list.pop(idx)
                    removed += 1
                changed = True
        return removed

    # Get valid swaps for removing codeword c
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c."""
        ball = ball_indices_np[c]
        unique = ball[cnt_arr[ball] == 1]

        if len(unique) == 0:
            # Any word not in code_set is valid
            return None

        # Intersection of balls of all unique words
        valid_set = set(ball_indices_np[unique[0]].tolist())
        for u in unique[1:]:
            valid_set &= set(ball_indices_np[u].tolist())
            if len(valid_set) == 0:
                return []

        # Filter out words already in code_set
        valid_list = [v for v in valid_set if v not in code_set]
        return valid_list

    # 2-swap: remove two codewords, add two others
    def get_valid_2_swaps(c1, c2, code_set, cnt_arr):
        """Return list of valid (c1_add, c2_add) pairs for removing c1 and c2."""
        ball1 = ball_indices_np[c1]
        ball2 = ball_indices_np[c2]
        
        # Words uniquely covered by c1 or c2
        unique1 = ball1[cnt_arr[ball1] == 1]
        unique2 = ball2[cnt_arr[ball2] == 1]
        
        # All unique words that need coverage
        all_unique = set(unique1.tolist()) | set(unique2.tolist())
        
        if len(all_unique) == 0:
            return None  # Any non-code words work
        
        # For each unique word, find candidate replacements
        # A replacement must cover all unique words
        candidates = set(range(N))
        for u in all_unique:
            candidates &= set(ball_indices_np[u].tolist())
            if len(candidates) == 0:
                return []
        
        # Filter out code words and the two being removed
        candidates -= code_set
        candidates.discard(c1)
        candidates.discard(c2)
        
        valid_list = list(candidates)
        return valid_list

    # Main search
    best_code = None
    best_size = N

    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_build(seed=None)
    if not unc:
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
        # Run delete phase until no more removals
        while run_delete_phase(code, cnt) > 0:
            pass
        best_code = code.copy()
        best_size = len(code)

    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 20.0:
        attempts += 1
        code, unc = greedy_build(seed=attempts)
        if not unc:
            cnt = np.zeros(N, dtype=np.int32)
            for c in code:
                cnt[ball_indices_np[c]] += 1

            # Local search loop
            no_improve = 0
            while time.time() < deadline - 10.0 and no_improve < 40:
                # Delete phase
                removed = run_delete_phase(code, cnt)
                if removed > 0:
                    no_improve = 0
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                else:
                    no_improve += 1

                # 1-swap phase
                if time.time() > deadline - 10.0:
                    break

                code_set = set(code)
                swaps_attempted = 0
                max_swaps = 300

                while swaps_attempted < max_swaps and time.time() < deadline - 10.0:
                    swaps_attempted += 1
                    if not code:
                        break

                    # Pick random codeword to remove
                    c = random.choice(code)
                    valid_swaps = get_valid_swaps(c, code_set, cnt)

                    if valid_swaps is None:
                        # Any non-code word works
                        c_prime = random.randint(0, N - 1)
                        while c_prime in code_set:
                            c_prime = random.randint(0, N - 1)
                    elif len(valid_swaps) == 0:
                        continue
                    else:
                        c_prime = random.choice(valid_swaps)

                    # Perform swap
                    code.remove(c)
                    code.append(c_prime)
                    code_set.remove(c)
                    code_set.add(c_prime)
                    cnt[ball_indices_np[c]] -= 1
                    cnt[ball_indices_np[c_prime]] += 1

                    # After swap, run delete phase
                    removed = run_delete_phase(code, cnt)
                    if removed > 0:
                        no_improve = 0
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                    else:
                        no_improve += 1

                # 2-swap phase (less frequent, more expensive)
                if time.time() > deadline - 10.0:
                    break
                if no_improve > 5:  # Only try 2-swaps when stuck
                    code_set = set(code)
                    two_swap_attempts = 0
                    max_two_swaps = 50

                    while two_swap_attempts < max_two_swaps and time.time() < deadline - 10.0:
                        two_swap_attempts += 1
                        if len(code) < 2:
                            break

                        # Pick two random codewords to remove
                        c1_idx = random.randint(0, len(code) - 1)
                        c2_idx = random.randint(0, len(code) - 1)
                        if c1_idx == c2_idx:
                            continue
                        c1 = code[c1_idx]
                        c2 = code[c2_idx]

                        valid_2_swaps = get_valid_2_swaps(c1, c2, code_set, cnt)

                        if valid_2_swaps is None:
                            # Any non-code words work
                            c1_prime = random.randint(0, N - 1)
                            while c1_prime in code_set:
                                c1_prime = random.randint(0, N - 1)
                            c2_prime = random.randint(0, N - 1)
                            while c2_prime in code_set or c2_prime == c1_prime:
                                c2_prime = random.randint(0, N - 1)
                        elif len(valid_2_swaps) < 2:
                            continue
                        else:
                            c1_prime, c2_prime = random.sample(valid_2_swaps, 2)

                        # Perform 2-swap
                        code.remove(c1)
                        code.remove(c2)
                        code.append(c1_prime)
                        code.append(c2_prime)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(c1_prime)
                        code_set.add(c2_prime)
                        cnt[ball_indices_np[c1]] -= 1
                        cnt[ball_indices_np[c2]] -= 1
                        cnt[ball_indices_np[c1_prime]] += 1
                        cnt[ball_indices_np[c2_prime]] += 1

                        # After 2-swap, run delete phase
                        removed = run_delete_phase(code, cnt)
                        if removed > 0:
                            no_improve = 0
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code.copy()
                        else:
                            no_improve += 1

        if best_size <= 186:
            break

    # Final polish: try more aggressive local search on best_code if time remains
    if best_code is not None and time.time() < deadline - 5.0:
        code = best_code.copy()
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
        code_set = set(code)
        
        polish_no_improve = 0
        while time.time() < deadline - 3.0 and polish_no_improve < 60:
            # Delete phase
            removed = run_delete_phase(code, cnt)
            if removed > 0:
                polish_no_improve = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
            else:
                polish_no_improve += 1
            
            # 1-swap phase with more attempts
            if time.time() > deadline - 3.0:
                break
            
            swaps_attempted = 0
            max_swaps = 400
            while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
                swaps_attempted += 1
                if not code:
                    break
                
                c = random.choice(code)
                valid_swaps = get_valid_swaps(c, code_set, cnt)
                
                if valid_swaps is None:
                    c_prime = random.randint(0, N - 1)
                    while c_prime in code_set:
                        c_prime = random.randint(0, N - 1)
                elif len(valid_swaps) == 0:
                    continue
                else:
                    c_prime = random.choice(valid_swaps)
                
                code.remove(c)
                code.append(c_prime)
                code_set.remove(c)
                code_set.add(c_prime)
                cnt[ball_indices_np[c]] -= 1
                cnt[ball_indices_np[c_prime]] += 1
                
                removed = run_delete_phase(code, cnt)
                if removed > 0:
                    polish_no_improve = 0
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                else:
                    polish_no_improve += 1
            
            # 2-swap phase in polish
            if time.time() > deadline - 3.0:
                break
            if polish_no_improve > 3:
                two_swap_attempts = 0
                max_two_swaps = 80
                while two_swap_attempts < max_two_swaps and time.time() < deadline - 3.0:
                    two_swap_attempts += 1
                    if len(code) < 2:
                        break

                    c1_idx = random.randint(0, len(code) - 1)
                    c2_idx = random.randint(0, len(code) - 1)
                    if c1_idx == c2_idx:
                        continue
                    c1 = code[c1_idx]
                    c2 = code[c2_idx]

                    valid_2_swaps = get_valid_2_swaps(c1, c2, code_set, cnt)

                    if valid_2_swaps is None:
                        c1_prime = random.randint(0, N - 1)
                        while c1_prime in code_set:
                            c1_prime = random.randint(0, N - 1)
                        c2_prime = random.randint(0, N - 1)
                        while c2_prime in code_set or c2_prime == c1_prime:
                            c2_prime = random.randint(0, N - 1)
                    elif len(valid_2_swaps) < 2:
                        continue
                    else:
                        c1_prime, c2_prime = random.sample(valid_2_swaps, 2)

                    code.remove(c1)
                    code.remove(c2)
                    code.append(c1_prime)
                    code.append(c2_prime)
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(c1_prime)
                    code_set.add(c2_prime)
                    cnt[ball_indices_np[c1]] -= 1
                    cnt[ball_indices_np[c2]] -= 1
                    cnt[ball_indices_np[c1_prime]] += 1
                    cnt[ball_indices_np[c2_prime]] += 1

                    removed = run_delete_phase(code, cnt)
                    if removed > 0:
                        polish_no_improve = 0
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                    else:
                        polish_no_improve += 1

    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_build(seed=None)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)

    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * 7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)

    return result


def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)

    # Simple greedy
    uncovered = set(range(total))
    selected = []

    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i

        if best_idx == -1:
            break

        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)

    # Ensure validity
    if uncovered:
        for j in uncovered:
            selected.append(j)

    return [list(all_words[i]) for i in selected]
