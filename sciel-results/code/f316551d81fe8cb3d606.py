"""Rewritten program for Egyptian fraction of 5/31 minimizing largest denominator."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    deadline = time.time() + 55  # safety margin before 60s kill
    target = Fraction(5, 31)
    best = None
    best_max = None

    def try_complete(rem, used, depth=0):
        nonlocal best, best_max
        if time.time() > deadline:
            return False
        if rem == 0:
            if not used:
                return False
            mx = max(used)
            if best_max is None or mx < best_max:
                best = list(used)
                best_max = mx
            return True
        if rem < 0 or len(used) >= 256:
            return False
        # Lower bound on next denominator: 1/d <= rem => d >= 1/rem
        # Upper bound: we want to keep max small, but also d must be > all used for distinctness ordering
        # Actually denominators just need to be distinct, not sorted, but we can WLOG sort them
        min_d = max(used[-1] + 1, 1) if used else 1
        # 1/d >= rem is impossible if rem > 1, but rem < 1 always here
        # d must satisfy 1/d <= rem => d >= ceil(1/rem)
        if rem.numerator > 0:
            min_d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        # Also d must be at least 2 if min_d < 2
        min_d = max(min_d, 2)
        # Upper bound for d: 1/d >= rem/256 roughly, but let's bound by trying reasonable range
        # For efficiency, limit search width
        max_d = min(10000000, min_d + 5000)
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return False
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            if new_rem == 0:
                used.append(d)
                mx = max(used)
                if best_max is None or mx < best_max:
                    best = list(used)
                    best_max = mx
                used.pop()
                continue
            used.append(d)
            if try_complete(new_rem, used, depth + 1):
                used.pop()
                return True
            used.pop()
        return False

    # Try greedy first for a quick baseline
    r = target
    greedy = []
    while r > 0 and len(greedy) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy.append(d)
        r -= Fraction(1, d)
    if r == 0 and greedy:
        best = list(greedy)
        best_max = max(greedy)

    # Now try to improve by searching from small denominators
    # Try different starting denominators
    for start in range(2, 20):
        if time.time() > deadline:
            break
        used = [start]
        rem = target - Fraction(1, start)
        if rem < 0:
            continue
        if try_complete(rem, used):
            if best_max is not None and best_max <= start:
                break

    if best is None:
        # Fallback: return greedy
        return greedy

    return best
# EVOLVE-BLOCK-END
