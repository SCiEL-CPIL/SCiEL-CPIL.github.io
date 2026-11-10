import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s kill

    def eval_sign_pattern(support, signs):
        """Compute nonzero count in P^2 for P with given support and signs (all mag=1)."""
        n = len(support)
        max_sum = support[-1] + support[-1]
        coeff = [0] * (max_sum + 1)
        for i in range(n):
            for j in range(i, n):
                s_val = signs[i] * signs[j]
                if i == j:
                    coeff[support[i] + support[j]] += s_val
                else:
                    coeff[support[i] + support[j]] += 2 * s_val
        return sum(1 for c in coeff if c != 0)

    best_score = 999
    found_best = None

    # Phase 1: Try consecutive supports {base, base+1, ..., base+11} with all 4096 sign patterns
    for base in range(30):
        if time.time() > deadline - 10:
            break
        support = tuple(range(base, base + 12))
        for mask in range(4096):
            signs = tuple(1 if (mask >> i) & 1 else -1 for i in range(12))
            score = eval_sign_pattern(support, signs)
            if score < best_score:
                best_score = score
                found_best = (support, signs)

    # Phase 2: Full enumeration of all 12-combinations, filter by sumset size
    if time.time() < deadline - 15:
        small_supports = []
        for combo in combinations(range(41), 12):
            if time.time() > deadline - 15:
                break
            c = list(combo)
            s = set()
            for i in range(12):
                for j in range(i, 12):
                    s.add(c[i] + c[j])
            ss = len(s)
            if ss <= 24:
                small_supports.append((ss, combo))

        for ss, combo in small_supports:
            if time.time() > deadline - 10:
                break
            if ss >= best_score:
                continue
            support = tuple(combo)
            for mask in range(4096):
                signs = tuple(1 if (mask >> i) & 1 else -1 for i in range(12))
                score = eval_sign_pattern(support, signs)
                if score < best_score:
                    best_score = score
                    found_best = (support, signs)

    # Phase 3: Structured random supports with small gaps
    if time.time() < deadline - 5:
        seen = set()
        for _ in range(100000):
            if time.time() > deadline - 5:
                break
            pos = 0
            support = [0]
            for _ in range(11):
                pos += 1 if random.random() < 0.8 else 2
                if pos > 40:
                    break
                support.append(pos)
            if len(support) == 12 and support[-1] <= 40:
                key = tuple(support)
                if key not in seen:
                    seen.add(key)
                    c = list(support)
                    s = set()
                    for i in range(12):
                        for j in range(i, 12):
                            s.add(c[i] + c[j])
                    if len(s) <= best_score + 1:
                        for mask in range(4096):
                            signs = tuple(1 if (mask >> i) & 1 else -1 for i in range(12))
                            score = eval_sign_pattern(support, signs)
                            if score < best_score:
                                best_score = score
                                found_best = (tuple(support), signs)

    # Build the final polynomial
    if found_best is not None:
        support, signs = found_best
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for i in range(12):
            coeffs[support[i]] = signs[i]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
    else:
        coeffs = [1] * 12

    # Validate
    if len(coeffs) > 41 or coeffs[-1] == 0 or sum(1 for c in coeffs if c != 0) != 12:
        coeffs = [1] * 12
    if any(c < -64 or c > 64 for c in coeffs):
        coeffs = [1] * 12

    return coeffs
