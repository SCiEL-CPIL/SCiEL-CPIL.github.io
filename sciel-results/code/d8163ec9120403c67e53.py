import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that 1 + 2*d <= 512 => d <= 255.5

    def time_ok():
        return time.time() < deadline

    # Bitmask representation: bit (v-1) set means v is in the set
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def delta_replace(mask, old_val, new_val):
        """
        Compute score delta when replacing old_val with new_val in mask.
        Returns (delta, new_mask).
        """
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        new_mask = (mask & ~old_bit) | new_bit

        # Collect all d values that could be affected by removing old_val or adding new_val
        affected_ds = set()

        for d in range(1, MAX_D + 1):
            # Check if old_val participates in a 3-AP with difference d
            # old_val as first: old_val, old_val+d, old_val+2d
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as middle: old_val-d, old_val, old_val+d
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as last: old_val-2d, old_val-d, old_val
            if old_val - 2 * d >= 1:
                affected_ds.add(d)

        for d in range(1, MAX_D + 1):
            # Check if new_val participates in a 3-AP with difference d
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)

        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has and not new_has:
                delta -= 1
            elif not old_has and new_has:
                delta += 1

        return delta, new_mask

    def local_search(mask, score, s_set, max_rounds=100, samples_per_swap=20):
        """
        Local search: repeatedly try to replace one element with another
        to improve the score.
        """
        lst = mask_to_list(mask)

        for round_idx in range(max_rounds):
            if not time_ok():
                break

            improved = False
            for _ in range(len(lst)):
                i = random.randint(0, N - 1)
                old_val = lst[i]

                best_new = old_val
                best_delta = 0
                best_new_mask = mask

                for _ in range(samples_per_swap):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in s_set:
                        continue

                    delta, new_mask = delta_replace(mask, old_val, new_val)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_val
                        best_new_mask = new_mask

                if best_new != old_val:
                    s_set.discard(old_val)
                    s_set.add(best_new)
                    lst[i] = best_new
                    mask = best_new_mask
                    score += best_delta
                    improved = True

            if not improved:
                break

        return mask, score, s_set, lst

    def simulated_annealing(mask, score, s_set, start_temp=10.0, end_temp=0.1,
                            max_iters=5000, samples_per_swap=15):
        """
        Simulated annealing to escape local optima.
        """
        import math

        lst = mask_to_list(mask)
        best_mask = mask
        best_score = score
        best_set = set(s_set)

        for it in range(max_iters):
            if not time_ok():
                break

            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)

            i = random.randint(0, N - 1)
            old_val = lst[i]

            best_new = old_val
            best_delta = 0
            best_new_mask = mask

            for _ in range(samples_per_swap):
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in s_set:
                    continue

                delta, new_mask = delta_replace(mask, old_val, new_val)
                if delta > best_delta:
                    best_delta = delta
                    best_new = new_val
                    best_new_mask = new_mask

            if best_new != old_val:
                # Accept or reject based on temperature
                if best_delta > 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(best_delta / temp)

                if accept:
                    s_set.discard(old_val)
                    s_set.add(best_new)
                    lst[i] = best_new
                    mask = best_new_mask
                    score += best_delta

                    if score > best_score:
                        best_score = score
                        best_mask = mask
                        best_set = set(s_set)

        return best_mask, best_score, best_set

    def greedy_construct():
        """
        Greedy construction: add one element at a time, choosing the one
        that maximizes the score.
        """
        mask = 0
        score = 0
        s_set = set()

        for _ in range(N):
            best_new = None
            best_s = score

            # Try a sample of candidates
            candidates = random.sample(range(1, MAX_VAL + 1), min(150, MAX_VAL))
            for v in candidates:
                if v in s_set:
                    continue
                new_mask = mask | (1 << (v - 1))
                new_s = score_mask(new_mask)
                if new_s > best_s:
                    best_s = new_s
                    best_new = v

            if best_new is None:
                best_new = random.choice(candidates)
                new_mask = mask | (1 << (best_new - 1))
                best_s = score_mask(new_mask)

            mask = new_mask
            score = best_s
            s_set.add(best_new)

        return mask, score, s_set

    # Main search
    random.seed(42)

    best_mask = 0
    best_score = 0
    best_list = []

    # Strategy 1: Greedy construction
    if time_ok():
        mask, score, s_set = greedy_construct()
        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = mask_to_list(mask)

        # Local search from greedy
        mask, score, s_set, lst = local_search(mask, score, s_set, max_rounds=50, samples_per_swap=15)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = lst[:]

    # Strategy 2: Interval [1, 32] start
    if time_ok():
        mask = 0
        for i in range(1, 33):
            mask |= (1 << (i - 1))
        score = score_mask(mask)
        s_set = set(range(1, 33))

        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = mask_to_list(mask)

        mask, score, s_set, lst = local_search(mask, score, s_set, max_rounds=50, samples_per_swap=15)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = lst[:]

    # Strategy 3: Arithmetic progression starts with various steps
    for step in [8, 12, 16, 20, 25]:
        if not time_ok():
            break
        lst = [1 + i * step for i in range(N)]
        if max(lst) > MAX_VAL:
            continue
        mask = list_to_mask(lst)
        score = score_mask(mask)
        s_set = set(lst)

        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = lst[:]

        mask, score, s_set, lst = local_search(mask, score, s_set, max_rounds=40, samples_per_swap=15)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = lst[:]

    # Strategy 4: Simulated annealing from the best found so far
    if time_ok():
        s_set = set(best_list)
        mask2, score2, s_set2 = simulated_annealing(best_mask, best_score, s_set,
                                                     start_temp=8.0, end_temp=0.05,
                                                     max_iters=3000, samples_per_swap=12)
        if score2 > best_score:
            best_score = score2
            best_mask = mask2
            best_list = mask_to_list(mask2)

        # Local search from SA result
        mask2, score2, s_set2, lst2 = local_search(mask2, score2, s_set2, max_rounds=50, samples_per_swap=15)
        if score2 > best_score:
            best_score = score2
            best_mask = mask2
            best_list = lst2[:]

    # Strategy 5: More random restarts with local search
    while time_ok():
        # Random initial set
        init_list = random.sample(range(1, MAX_VAL + 1), N)
        mask = list_to_mask(init_list)
        score = score_mask(mask)
        s_set = set(init_list)

        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = init_list[:]

        # Short local search
        mask, score, s_set, lst = local_search(mask, score, s_set, max_rounds=30, samples_per_swap=12)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_list = lst[:]

    return sorted(best_list)
