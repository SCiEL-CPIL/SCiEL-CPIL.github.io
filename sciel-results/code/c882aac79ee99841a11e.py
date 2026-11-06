"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _two_term_solutions(a, b, min_d, max_d):
    """Find all (x,y) with min_d<=x<y<=max_d, 1/x+1/y = a/b."""
    if a <= 0 or b <= 0:
        return []
    a2 = a * a
    sols = []
    i = 1
    while i * i <= a2:
        if a2 % i == 0:
            d1 = i
            d2 = a2 // i
            for d, e in [(d1, d2), (d2, d1)]:
                num_x = a + d
                num_y = a + e
                if num_x % b == 0 and num_y % b == 0:
                    x = num_x // b
                    y = num_y // b
                    if min_d <= x < y <= max_d:
                        sols.append((x, y))
        i += 1
    return sols

def _dfs(rem_num, rem_den, min_d, max_d, used, path, best, deadline):
    if time.time() > deadline:
        return
    
    if rem_num == 0:
        if path:
            m = max(path)
            if m < best[0]:
                best[0] = m
                best[1] = list(path)
        return
    
    if rem_num < 0:
        return
    
    # Prune: if rem < 1/max_d, can't represent
    if rem_num * max_d < rem_den:
        return
    
    # Try two-term completion
    sols = _two_term_solutions(rem_num, rem_den, min_d, max_d)
    for (x, y) in sols:
        if x not in used and y not in used:
            path.append(x)
            path.append(y)
            used.add(x)
            used.add(y)
            m = max(path)
            if m < best[0]:
                best[0] = m
                best[1] = list(path)
            used.remove(x)
            used.remove(y)
            path.pop()
            path.pop()
            if time.time() > deadline:
                return
    
    # Try one more term
    d_min = max(min_d, (rem_den + rem_num - 1) // rem_num)
    if d_min > max_d:
        return
    
    for d in range(d_min, max_d + 1):
        if d in used:
            continue
        if d * rem_num < rem_den:
            continue
        new_num = rem_num * d - rem_den
        new_den = rem_den * d
        if new_num < 0:
            continue
        if new_num == 0:
            path.append(d)
            used.add(d)
            m = max(path)
            if m < best[0]:
                best[0] = m
                best[1] = list(path)
            used.remove(d)
            path.pop()
        else:
            path.append(d)
            used.add(d)
            _dfs(new_num, new_den, d + 1, max_d, used, path, best, deadline)
            used.remove(d)
            path.pop()
        if time.time() > deadline:
            return

def construct():
    target = Fraction(5, 31)
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Greedy fallback
    r = target
    greedy_vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    greedy_max = max(greedy_vals)
    
    best = [greedy_max, greedy_vals]
    
    # Search
    for d1 in range(7, 500):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, d1)
        if rem <= 0:
            continue
        # Two-term
        sols = _two_term_solutions(rem.numerator, rem.denominator, d1 + 1, best[0] - 1)
        for (x, y) in sols:
            if x != d1 and y != d1:
                m = max(d1, x, y)
                if m < best[0]:
                    best[0] = m
                    best[1] = [d1, x, y]
        # DFS
        if best[0] > d1 + 1:
            used = {d1}
            path = [d1]
            _dfs(rem.numerator, rem.denominator, d1 + 1, best[0] - 1, used, path, best, deadline)
    
    return best[1]
# EVOLVE-BLOCK-END
