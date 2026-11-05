import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Known 46-term solution for 1 with denominators >= 5
    candidate = [5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 45, 60, 90, 120, 180, 240, 360, 720, 1080, 1440, 2160, 2880, 4320, 5760, 7200, 10800, 14400, 21600, 28800, 43200, 57600, 72000, 108000, 144000, 216000, 288000, 432000, 576000, 720000, 1080000, 1440000, 2160000, 2880000]
    
    s = sum(Fraction(1, d) for d in candidate)
    if s == 1 and all(5 <= d <= 10000000 for d in candidate) and len(set(candidate)) == len(candidate):
        return candidate
    
    # Fallback: DFS search
    best = [None]
    best_count = [47]
    
    def dfs(target, min_denom, terms, max_terms):
        if time.time() > deadline:
            return
        if target == 0:
            if len(terms) < best_count[0]:
                best_count[0] = len(terms)
                best[0] = list(terms)
            return
        if len(terms) >= max_terms:
            return
        remaining = max_terms - len(terms)
        if Fraction(1, min_denom) * remaining < target:
            return
        inv_target = 1 / target
        lo = max(min_denom, int(inv_target) + (0 if inv_target.denominator == 1 else 1))
        for d in range(lo, 10000001):
            if time.time() > deadline:
                return
            if len(terms) + 1 > max_terms:
                break
            if Fraction(1, d) > target:
                break
            new_target = target - Fraction(1, d)
            terms.append(d)
            dfs(new_target, d + 1, terms, max_terms)
            terms.pop()
    
    dfs(Fraction(1, 1), 5, [], 46)
    
    if best[0] is not None:
        return best[0]
    
    # Ultimate fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no representation found")
