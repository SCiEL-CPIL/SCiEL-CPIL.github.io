import time
import itertools

def construct():
    start_time = time.time()
    LIMIT = 55.0
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    N = len(smooth_numbers)
    
    # Precompute divisibility relationships for smooth numbers
    # divs[i] is a set of indices j such that smooth_numbers[j] divides smooth_numbers[i]
    # Since we need antichains, we want to avoid pairs where one divides the other.
    # Let's precompute a bitmask or a list of "incompatible" indices for each smooth number.
    # incompatible[i] is a set of indices j (j != i) such that smooth_numbers[i] divides smooth_numbers[j] or vice versa.
    
    incompatible = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if b % a == 0 or a % b == 0:
                incompatible[i].add(j)
                incompatible[j].add(i)
                
    # 2. Helper to check if a list of indices forms an antichain
    def is_antichain_indices(indices):
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                if indices[j] in incompatible[indices[i]]:
                    return False
        return True

    # 3. Pre-compute best representation for each target 1..128
    # answers[n] will store the list of terms for target n+1
    answers = [None] * 128
    
    # Level 1: Single term
    for i, s in enumerate(smooth_numbers):
        if s <= 128:
            answers[s - 1] = [s]

    # Level 2: Two terms
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if answers[s - 1] is None:
                # Check antichain
                if j not in incompatible[i]:
                    answers[s - 1] = [a, b]

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 3: Three terms
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if j in incompatible[i]:
                continue
            if a + b > 128:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s = a + b + c
                if s > 128:
                    break
                if answers[s - 1] is None:
                    # Check antichain for (a, b, c)
                    if k not in incompatible[i] and k not in incompatible[j]:
                        answers[s - 1] = [a, b, c]

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 4: Four terms
    # Optimization: Pre-check pairs (i, j) that are compatible.
    # Then for each compatible pair (i, j), iterate k > j compatible with both.
    # Then iterate l > k compatible with all.
    
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if j in incompatible[i]:
                continue
            if a + b > 128:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                if a + b + c > 128:
                    break
                for l in range(k + 1, N):
                    d = smooth_numbers[l]
                    s = a + b + c + d
                    if s > 128:
                        break
                    if l in incompatible[i] or l in incompatible[j] or l in incompatible[k]:
                        continue
                    if answers[s - 1] is None:
                        answers[s - 1] = [a, b, c, d]

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 5: Five terms
    # This might be slow, so add time check
    if time.time() - start_time > LIMIT:
        # Fallback: try to find any valid representation for missing targets using brute force with time limit
        for i in range(128):
            if answers[i] is None:
                # Use itertools.combinations but stop if time is up
                target = i + 1
                found = False
                for r in range(5, 10):
                    if time.time() - start_time > LIMIT:
                        break
                    for combo in itertools.combinations(smooth_numbers, r):
                        if sum(combo) == target:
                            if is_antichain_indices(list(combo)):
                                answers[i] = list(combo)
                                found = True
                                break
                    if found:
                        break
        return answers

    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if j in incompatible[i]:
                continue
            if a + b > 128:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                if a + b + c > 128:
                    break
                for l in range(k + 1, N):
                    d = smooth_numbers[l]
                    if l in incompatible[i] or l in incompatible[j] or l in incompatible[k]:
                        continue
                    s4 = a + b + c + d
                    if s4 > 128:
                        break
                    for m in range(l + 1, N):
                        e = smooth_numbers[m]
                        s = s4 + e
                        if s > 128:
                            break
                        if m in incompatible[i] or m in incompatible[j] or m in incompatible[k] or m in incompatible[l]:
                            continue
                        if answers[s - 1] is None:
                            answers[s - 1] = [a, b, c, d, e]
                            # Break innermost if we just want one? No, s changes.
                            # But if we found one for s, we can break the 'for m' loop?
                            # No, because the next m gives a different s.
                            # However, if we are looking for a specific target, we could break.
                            # But we are filling all targets.
                            # So we continue.

    # Final verification
    for i in range(128):
        if answers[i] is None:
            # Fallback
            target = i + 1
            # Try up to 7 terms
            for r in range(5, 8):
                if time.time() - start_time > LIMIT:
                    break
                for combo in itertools.combinations(smooth_numbers, r):
                    if sum(combo) == target and is_antichain_indices(list(combo)):
                        answers[i] = list(combo)
                        break
                if answers[i] is not None:
                    break

    return answers
