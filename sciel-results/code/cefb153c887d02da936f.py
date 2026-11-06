def construct():
    import time
    from fractions import Fraction
    deadline = time.time() + 50
    target = Fraction(5, 31)
    best_val = 10**9
    best_sol = None

    def dfs(rem, last, current, depth):
        nonlocal best_val, best_sol
        if time.time() > deadline:
            return
        if rem == 0:
            if current and max(current) < best_val:
                best_val = max(current)
                best_sol = list(current)
            return
        if rem < 0 or depth >= 7:
            return
        lo = (rem.denominator + rem.numerator - 1) // rem.numerator
        if lo <= last:
            lo = last + 1
        if lo >= best_val:
            return
        hi = min(lo + 60, best_val - 1, 10**7)
        for k in range(lo, hi + 1):
            nr = rem - Fraction(1, k)
            if nr < 0:
                break
            current.append(k)
            dfs(nr, k, current, depth + 1)
            current.pop()
            if time.time() > deadline:
                return

    # seed with greedy
    def greedy(r):
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals
    g = greedy(target)
    best_val = max(g)
    best_sol = g

    dfs(target, 0, [], 0)
    return best_sol
