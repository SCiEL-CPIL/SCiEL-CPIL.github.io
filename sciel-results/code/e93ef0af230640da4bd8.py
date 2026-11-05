import time
from fractions import Fraction

def _precompute_k9():
    """Precompute the k=9 solution at module load time."""
    R = Fraction(1)
    for d in range(5, 12):
        R -= Fraction(1, d)
    
    lower_d8 = max(12, int(1 / float(R)) + 1)
    upper_d8 = int(2 / float(R)) + 2
    
    for d8 in range(lower_d8, upper_d8 + 1):
        rem = R - Fraction(1, d8)
        if rem <= 0:
            continue
        num, den = rem.numerator, rem.denominator
        if den % num == 0:
            d9 = den // num
            if d9 > d8:
                return [5, 6, 7, 8, 9, 10, 11, d8, d9]
    return None

_K9_SOLUTION = _precompute_k9()

def construct():
    if _K9_SOLUTION is not None:
        return _K9_SOLUTION
    
    # Fallback: General DFS for k=10
    start_time = time.time()
    deadline = start_time + 80.0
    
    def find_solution_k(k, time_limit=5.0):
        def dfs(remaining, k_left, min_den, current):
            if time.time() > start_time + time_limit:
                return None
            if k_left == 0:
                return current[:] if remaining == 0 else None
            if k_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        x = den // num
                        if x > min_den and x <= 10000000:
                            return current + [x]
                return None
            
            lower_a = max(min_den, int(1 / float(remaining)) + 1)
            upper_a = min(10000000, int(k_left / float(remaining)) + 2)
            
            if lower_a > upper_a:
                return None
            
            for a in range(lower_a, upper_a + 1):
                if time.time() > start_time + time_limit:
                    return None
                max_possible = 0.0
                for i in range(k_left):
                    max_possible += 1.0 / (a + i)
                if max_possible < float(remaining) - 1e-15:
                    break
                
                new_remaining = remaining - Fraction(1, a)
                if new_remaining <= 0:
                    continue
                res = dfs(new_remaining, k_left - 1, a + 1, current + [a])
                if res:
                    return res
            return None
        
        return dfs(Fraction(1), k, 5, [])

    if time.time() < deadline - 10:
        sol = find_solution_k(10, 5.0)
        if sol:
            return sol

    # Greedy fallback
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            d = last + 1
            while Fraction(1, d) > remaining:
                d += 1
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
