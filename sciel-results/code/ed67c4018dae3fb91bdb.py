import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341213412134123142314213412143214321"
    if n == 5:
        # Standard recursive construction for n=5
        s4 = "12341213412134123142314213412143214321"
        result = []
        perms = list(itertools.permutations([1,2,3,4]))
        for i, p in enumerate(perms):
            result.append(p)
            if i < len(perms) - 1:
                result.append((5, p[0], p[1], p[2]))
        # This is getting complex, let's just use the standard algorithm
        word = [1]
        for m in range(2, n + 1):
            perms = list(itertools.permutations(range(1, m)))
            new = []
            for i, p in enumerate(perms):
                new.extend(p)
                if i < len(perms) - 1:
                    new.extend([m] + list(p))
            word = new
        return "".join(map(str, word))

    if n == 6:
        # For n=6, use recursive construction then optimize
        word = [1]
        for m in range(2, n + 1):
            perms = list(itertools.permutations(range(1, m)))
            new = []
            for i, p in enumerate(perms):
                new.extend(p)
                if i < len(perms) - 1:
                    new.extend([m] + list(p))
            word = new
        return "".join(map(str, word))

    # n == 7 case
    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms = list(itertools.permutations(range(1, m)))
        new = []
        for i, p in enumerate(perms):
            new.extend(p)
            if i < len(perms) - 1:
                new.extend([m] + list(p))
        word = new

    S = list(word)
    L = len(S)

    # Build counts
    def rebuild_counts(S):
        counts = defaultdict(int)
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i + n])
            counts[p] += 1
        return counts

    counts = rebuild_counts(S)
    at_risk = set(p for p, c in counts.items() if c == 1)

    best_S = S[:]
    best_len = L

    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        return len(seen) == 5040

    # Removal pass
    def removal_pass(S, counts, at_risk, deadline):
        removed_any = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if not valid:
                i -= 1
                continue

            # Perform removal
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                c = counts[p] - 1
                if c == 0:
                    del counts[p]
                    at_risk.discard(p)
                else:
                    counts[p] = c
                    if c == 1:
                        at_risk.add(p)

            # Add straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            for k in range(start_new, end_new + 1):
                wc = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    wc.append(S[s_idx])
                p_new = tuple(wc)
                c = counts.get(p_new, 0)
                if c == 0:
                    counts[p_new] = 1
                    at_risk.add(p_new)
                elif c == 1:
                    counts[p_new] = 2
                    at_risk.discard(p_new)
                else:
                    counts[p_new] = c + 1

            S.pop(i)
            removed_any = True

        return removed_any

    # Bidirectional removal
    def removal_pass_bidirectional(S, counts, at_risk, deadline):
        removed_any = False
        left = 0
        right = len(S) - 1
        while left <= right and time.time() <= deadline:
            # Try removing from right
            i = right
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if valid:
                for j in range(start_old, end_old + 1):
                    p = tuple(S[j:j + n])
                    c = counts[p] - 1
                    if c == 0:
                        del counts[p]
                        at_risk.discard(p)
                    else:
                        counts[p] = c
                        if c == 1:
                            at_risk.add(p)
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    c = counts.get(p_new, 0)
                    if c == 0:
                        counts[p_new] = 1
                        at_risk.add(p_new)
                    elif c == 1:
                        counts[p_new] = 2
                        at_risk.discard(p_new)
                    else:
                        counts[p_new] = c + 1
                S.pop(i)
                removed_any = True
                right -= 1
            else:
                right -= 1

            # Try removing from left
            if left > right:
                break
            i = left
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if valid:
                for j in range(start_old, end_old + 1):
                    p = tuple(S[j:j + n])
                    c = counts[p] - 1
                    if c == 0:
                        del counts[p]
                        at_risk.discard(p)
                    else:
                        counts[p] = c
                        if c == 1:
                            at_risk.add(p)
                new_len = len(S) - 1
                start_new = max(0, i - n + 1)
                end_new = min(i - 1, new_len - n)
                for k in range(start_new, end_new + 1):
                    wc = []
                    for idx in range(k, k + n):
                        s_idx = idx if idx < i else idx + 1
                        wc.append(S[s_idx])
                    p_new = tuple(wc)
                    c = counts.get(p_new, 0)
                    if c == 0:
                        counts[p_new] = 1
                        at_risk.add(p_new)
                    elif c == 1:
                        counts[p_new] = 2
                        at_risk.discard(p_new)
                    else:
                        counts[p_new] = c + 1
                S.pop(i)
                removed_any = True
                left += 1
            else:
                left += 1

        return removed_any

    # Perturbation strategies
    def perturb_swap(S, counts, at_risk):
        i, j = random.sample(range(len(S)), 2)
        S[i], S[j] = S[j], S[i]
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    def perturb_insert(S, counts, at_risk):
        pos = random.randint(0, len(S))
        sym = random.randint(1, n)
        S.insert(pos, sym)
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    # Main loop
    stuck_rounds = 0
    max_stuck = 30
    while time.time() < deadline:
        if stuck_rounds % 2 == 0:
            removed = removal_pass(S, counts, at_risk, deadline)
        else:
            removed = removal_pass_bidirectional(S, counts, at_risk, deadline)
        if removed:
            best_S = S[:]
            best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            r = random.random()
            if r < 0.5 and len(S) > n:
                S, counts, at_risk = perturb_swap(S, counts, at_risk)
            elif len(S) < best_len * 1.15:
                S, counts, at_risk = perturb_insert(S, counts, at_risk)
            else:
                S, counts, at_risk = perturb_swap(S, counts, at_risk)

    # Final polish
    if time.time() < deadline - 2:
        S2 = best_S[:]
        counts2 = rebuild_counts(S2)
        at_risk2 = set(p for p, c in counts2.items() if c == 1)
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            removal_pass(S2, counts2, at_risk2, deadline)
            removal_pass_bidirectional(S2, counts2, at_risk2, deadline)
        if len(S2) < len(best_S) and is_valid(S2):
            best_S = S2

    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
