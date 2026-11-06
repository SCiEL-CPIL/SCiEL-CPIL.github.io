"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def _is_valid(subset, recips, recips_f):
    s_list = list(subset)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        target = recips[a]
        target_f = recips_f[a]
        others = [s_list[j] for j in range(n) if j != i]
        m = len(others)
        for mask in range(1, 1 << m):
            s_f = 0.0
            for j in range(m):
                if mask & (1 << j):
                    s_f += recips_f[others[j]]
            if abs(s_f - target_f) < 1e-10:
                s = Fraction(0, 1)
                for j in range(m):
                    if mask & (1 << j):
                        s += recips[others[j]]
                if s == target:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 80
    
    recips = {x: Fraction(1, x) for x in range(1, 25)}
    recips_f = {x: 1.0 / x for x in range(1, 25)}
    
    # Greedy in descending order
    result = []
    for e in range(24, 0, -1):
        candidate = result + [e]
        if _is_valid(candidate, recips, recips_f):
            result = candidate
    
    current_best = set(result)
    best_size = len(current_best)
    
    all_elements = set(range(1, 25))
    
    # Try to add elements
    for e in sorted(all_elements - current_best, reverse=True):
        if time.time() - start > deadline - 30:
            break
        candidate = current_best | {e}
        if _is_valid(list(candidate), recips, recips_f):
            current_best = candidate
            best_size = len(current_best)
    
    # Remove-1-add-2
    for _ in range(30):
        if time.time() - start > deadline - 10:
            break
        improved = False
        for a in list(current_best):
            for b in sorted(all_elements - current_best, reverse=True):
                for c in sorted(all_elements - current_best - {b}, reverse=True):
                    candidate = (current_best - {a}) | {b, c}
                    if len(candidate) > best_size and _is_valid(list(candidate), recips, recips_f):
                        current_best = candidate
                        best_size = len(current_best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final add
    for e in sorted(all_elements - current_best, reverse=True):
        if time.time() - start > deadline - 2:
            break
        candidate = current_best | {e}
        if _is_valid(list(candidate), recips, recips_f):
            current_best = candidate
            best_size = len(current_best)
    
    result = sorted(current_best)
    if not result:
        result = [24]
    return result
# EVOLVE-BLOCK-END
