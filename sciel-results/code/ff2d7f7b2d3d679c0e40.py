import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs
    
    # Find top divisor sets by size
    divisor_sets = []
    for L in range(1, N + 1):
        divs = get_divisors(L)
        divisor_sets.append((len(divs), L, divs))
    divisor_sets.sort(reverse=True)
    
    def can_add(x, s):
        for y in s:
            if x // gcd(x, y) * y > N:
                return False
        return True
    
    def greedy_expand(s):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            candidates = [x for x in range(1, N + 1) if x not in s]
            # Score by compatibility with existing set
            def score(x):
                compat = sum(1 for y in s if x // gcd(x, y) * y <= N)
                return (-compat, x)
            candidates.sort(key=score)
            for x in candidates:
                if can_add(x, s):
                    s.add(x)
                    changed = True
                    break
        return s
    
    # Try multiple seeds from top divisor sets
    best_result = set()
    best_size = 0
    for _, L, divs in divisor_sets[:5]:
        if time.time() > deadline - 5:
            break
        expanded = greedy_expand(set(divs))
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_result = expanded
    
    current_set = best_result
    current_size = best_size
    
    # Local search: remove one element, re-expand
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for rc in list(current_set):
            if time.time() > deadline:
                break
            trial = current_set - {rc}
            expanded = greedy_expand(set(trial))
            if len(expanded) > current_size:
                current_set = expanded
                current_size = len(expanded)
                improved = True
                break
    
    final = greedy_expand(set(current_set))
    return list(final)
