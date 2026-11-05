import time

def construct():
    start = time.time()
    deadline = start + 115.0
    full_mask = (1 << 30) - 1
    max_size = 32

    def is_valid_add(x, S, used_diffs_mask):
        for s in S:
            d = x - s
            if d <= 0:
                continue
            if (used_diffs_mask >> d) & 1:
                return False
        return True

    def add_to_mask(used_diffs_mask, x, S):
        new_mask = used_diffs_mask
        for s in S:
            d = x - s
            if d > 0:
                new_mask |= (1 << d)
        return new_mask

    def coverage_gain(x, S, covered_mask):
        gain = 0
        for s in S:
            d = x - s
            if 1 <= d <= 30:
                gain |= (1 << (d - 1))
        return gain

    def dfs(S, used_diffs_mask, covered_mask, D, target_size):
        if time.time() > deadline:
            return None

        if (covered_mask & full_mask) == full_mask:
            return list(S)

        if len(S) >= target_size:
            return None

        # Pruning: can we cover all missing diffs with remaining slots?
        missing = full_mask & ~covered_mask
        if missing == 0:
            return list(S)
        
        # If we've reached target size, we must have full coverage
        if len(S) == target_size:
            return None

        remaining_slots = target_size - len(S)
        if remaining_slots == 0:
            return None

        # Max new differences we can add
        # Each new element adds (len(S)) differences with existing ones.
        # Total new diffs from k new elements: k*len(S) + k*(k-1)//2
        max_new_diffs = remaining_slots * len(S) + remaining_slots * (remaining_slots - 1) // 2
        if bin(missing).count('1') > max_new_diffs:
            return None

        candidates = []
        for x in range(S[-1] + 1, D + 1):
            if not is_valid_add(x, S, used_diffs_mask):
                continue
            gain = coverage_gain(x, S, covered_mask)
            new_covered = covered_mask | gain
            missing_covered = gain & missing
            coverage_count = bin(missing_covered).count('1')
            candidates.append((-coverage_count, x, gain))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], t[1]))

        # Limit candidates
        limit = 50
        for _, x, gain in candidates[:limit]:
            new_mask = add_to_mask(used_diffs_mask, x, S)
            new_covered = covered_mask | gain
            S.append(x)
            res = dfs(S, new_mask, new_covered, D, target_size)
            S.pop()
            if res is not None:
                return res

        return None

    # Search for minimum D
    # We know D < 37 is impossible.
    for D in range(37, 100):
        if time.time() > deadline - 5:
            break
        
        # Determine valid sizes k for this D
        # k(k-1)/2 >= 30 => k >= 9
        # k(k-1)/2 <= D
        valid_sizes = []
        for k in range(9, max_size + 1):
            if k * (k - 1) // 2 <= D:
                valid_sizes.append(k)
            else:
                break
        
        if not valid_sizes:
            continue

        # Try smallest k first (tighter constraint, often faster to prove/disprove or find)
        # Actually, smaller k means fewer elements to place, but harder to cover 30 diffs?
        # For D=37, only k=9 is valid.
        for k in valid_sizes:
            res = dfs([0], 0, 0, D, k)
            if res is not None:
                return res

    # Fallback greedy
    S = [0]
    used_diffs_mask = 0
    covered_mask = 0
    
    while (covered_mask & full_mask) != full_mask and len(S) < max_size:
        if time.time() > deadline - 2:
            break
        best_x = None
        best_gain = 0
        for x in range(S[-1] + 1, 200):
            if not is_valid_add(x, S, used_diffs_mask):
                continue
            gain = coverage_gain(x, S, covered_mask)
            new_covered = covered_mask | gain
            missing_covered = gain & (full_mask & ~covered_mask)
            cov = bin(missing_covered).count('1')
            if cov > best_gain:
                best_gain = cov
                best_x = x
            elif cov == best_gain and best_x is not None and x < best_x:
                best_x = x

        if best_x is None:
            for x in range(S[-1] + 1, 200):
                if is_valid_add(x, S, used_diffs_mask):
                    best_x = x
                    break

        if best_x is None:
            break

        S.append(best_x)
        used_diffs_mask = add_to_mask(used_diffs_mask, best_x, S)
        covered_mask |= coverage_gain(best_x, S[:-1], covered_mask)

    return S
