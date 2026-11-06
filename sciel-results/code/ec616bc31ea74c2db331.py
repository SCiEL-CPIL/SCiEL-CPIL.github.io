import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
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
    
    # 2. Helper to check if a list of numbers forms an antichain
    def is_antichain(terms):
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = terms[i], terms[j]
                if a == b:
                    return False
                if b % a == 0 or a % b == 0:
                    return False
        return True
    
    # Precompute divisibility relation for speed
    # div[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    div = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and smooth_numbers[j] % smooth_numbers[i] == 0:
                div[i][j] = True
    
    def is_antichain_fast(terms):
        """Fast antichain check using precomputed div table"""
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                a_idx = smooth_numbers.index(terms[i]) if terms[i] in smooth_numbers else -1
                b_idx = smooth_numbers.index(terms[j]) if terms[j] in smooth_numbers else -1
                # Actually, let's just use direct check since we have the values
                a, b = terms[i], terms[j]
                if b % a == 0 or a % b == 0:
                    return False
        return True

    # 3. Pre-compute best representation for each target 1..128
    # answers[n] will store the list of terms for target n+1
    answers = [None] * 128
    
    # Level 1: Single term
    for s in smooth_numbers:
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
                if b % a != 0 and a % b != 0:
                    answers[s - 1] = [a, b]

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 3: Three terms
    for i in range(N):
        a = smooth_numbers[i]
        if not time_ok():
            break
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if a + b > 128:
                continue
            if not time_ok():
                break
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s = a + b + c
                if s > 128:
                    break
                if answers[s - 1] is None:
                    # Check antichain for (a, b, c)
                    if (b % a != 0 and a % b != 0) and \
                       (c % a != 0 and a % c != 0) and \
                       (c % b != 0 and b % c != 0):
                        answers[s - 1] = [a, b, c]
            if not time_ok():
                break

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 4: Four terms - more efficient approach
    # For each target, try to find 4-term antichain sums
    missing_targets = [i + 1 for i in range(128) if answers[i] is None]
    
    if not time_ok() or not missing_targets:
        if all(ans is not None for ans in answers):
            return answers
    
    # For 4-term search, iterate over pairs and try to complete
    # Use a more systematic approach: for each pair (a,b) that's an antichain,
    # find pairs (c,d) that sum to target - a - b and form antichain with (a,b)
    
    for target in missing_targets:
        if not time_ok():
            break
        found = False
        for i in range(N):
            if found or not time_ok():
                break
            a = smooth_numbers[i]
            if a >= target:
                break
            for j in range(i + 1, N):
                if found or not time_ok():
                    break
                b = smooth_numbers[j]
                if a + b >= target:
                    break
                # Check if a,b form antichain
                if b % a == 0 or a % b == 0:
                    continue
                remaining = target - a - b
                # Need c + d = remaining where c,d > b (to maintain ordering)
                # and all four form antichain
                for k in range(j + 1, N):
                    if found or not time_ok():
                        break
                    c = smooth_numbers[k]
                    if c >= remaining:
                        break
                    d = remaining - c
                    if d <= c:
                        continue
                    # Check if d is in smooth_numbers
                    if d not in smooth_numbers:
                        continue
                    l = smooth_numbers.index(d)
                    if l <= k:
                        continue
                    # Check antichain for all four
                    terms = [a, b, c, d]
                    antichain = True
                    for p in range(4):
                        for q in range(p + 1, 4):
                            if terms[q] % terms[p] == 0 or terms[p] % terms[q] == 0:
                                antichain = False
                                break
                        if not antichain:
                            break
                    if antichain:
                        answers[target - 1] = [a, b, c, d]
                        found = True
                        break

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Level 5: Five terms for remaining
    missing_targets = [i + 1 for i in range(128) if answers[i] is None]
    
    for target in missing_targets:
        if not time_ok():
            break
        found = False
        # Try to find 5-term antichain
        for i in range(N):
            if found or not time_ok():
                break
            a = smooth_numbers[i]
            if a >= target:
                break
            for j in range(i + 1, N):
                if found or not time_ok():
                    break
                b = smooth_numbers[j]
                if a + b >= target:
                    break
                if b % a == 0 or a % b == 0:
                    continue
                rem1 = target - a - b
                for k in range(j + 1, N):
                    if found or not time_ok():
                        break
                    c = smooth_numbers[k]
                    if c >= rem1:
                        break
                    if c % a == 0 or a % c == 0 or c % b == 0 or b % c == 0:
                        continue
                    rem2 = rem1 - c
                    for l in range(k + 1, N):
                        if found or not time_ok():
                            break
                        d = smooth_numbers[l]
                        if d >= rem2:
                            break
                        if d % a == 0 or a % d == 0 or d % b == 0 or b % d == 0 or d % c == 0 or c % d == 0:
                            continue
                        e = rem2 - d
                        if e <= d:
                            continue
                        if e not in smooth_numbers:
                            continue
                        m = smooth_numbers.index(e)
                        if m <= l:
                            continue
                        # Check full antichain
                        terms = [a, b, c, d, e]
                        antichain = True
                        for p in range(5):
                            for q in range(p + 1, 5):
                                if terms[q] % terms[p] == 0 or terms[p] % terms[q] == 0:
                                    antichain = False
                                    break
                            if not antichain:
                                break
                        if antichain:
                            answers[target - 1] = [a, b, c, d, e]
                            found = True
                            break

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers

    # Fallback: try up to 8 terms for any remaining
    missing_targets = [i + 1 for i in range(128) if answers[i] is None]
    
    for target in missing_targets:
        if not time_ok():
            break
        found = False
        # Use combinations but with pruning
        for r in range(6, 9):
            if found or not time_ok():
                break
            for combo in itertools.combinations(smooth_numbers, r):
                if sum(combo) == target and is_antichain(list(combo)):
                    answers[target - 1] = list(combo)
                    found = True
                    break

    # Final fallback: if any still missing, use greedy approach
    for i in range(128):
        if answers[i] is None:
            target = i + 1
            # Greedy: pick largest possible smooth numbers that don't divide each other
            remaining = target
            terms = []
            for s in sorted(smooth_numbers, reverse=True):
                if s > remaining:
                    continue
                # Check if s divides or is divided by any existing term
                compatible = True
                for t in terms:
                    if s % t == 0 or t % s == 0:
                        compatible = False
                        break
                if compatible:
                    terms.append(s)
                    remaining -= s
                    if remaining == 0:
                        break
            if remaining == 0 and len(terms) <= 30:
                answers[i] = terms
            else:
                # Last resort: just use 1s (but 1 divides everything, so this won't work)
                # Use a different approach: find any valid representation
                # For now, leave as None which will cause failure, but we should have found something
                pass

    return answers
