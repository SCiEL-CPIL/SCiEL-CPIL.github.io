"""Few-term Egyptian fractions with denominators > 4; minimize count."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def elapsed_ok():
        return time.time() < deadline

    best_solution = None

    def search(start_den, remaining, terms, current_terms):
        nonlocal best_solution
        if not elapsed_ok():
            return
        if terms == 0:
            if remaining == 0:
                if best_solution is None or len(current_terms) < len(best_solution):
                    best_solution = list(current_terms)
            return
        if remaining <= 0:
            return
        # Pruning: max possible sum
        max_possible = Fraction(0, 1)
        for i in range(terms):
            max_possible += Fraction(1, start_den + i)
        if max_possible < remaining:
            return
        # d must satisfy 1/d < remaining => d > 1/remaining
        # and d >= start_den
        d_min = start_den
        # 1/d <= remaining => d >= 1/remaining
        if remaining > 0:
            d_lower = int(1 / remaining) + 1
            if d_lower > d_min:
                d_min = d_lower
        # Upper bound: 1/d + (terms-1)/(d+1) >= remaining
        # For simplicity, d < some bound
        d_max = d_min + 200  # heuristic
        if d_max > 10000000:
            d_max = 10000000

        for d in range(d_min, d_max + 1):
            if not elapsed_ok():
                return
            frac = Fraction(1, d)
            if frac > remaining:
                break
            new_remaining = remaining - frac
            current_terms.append(d)
            search(d + 1, new_remaining, terms - 1, current_terms)
            current_terms.pop()

    # Try k from 8 to 12 (we proved k<8 impossible)
    for k in range(8, 13):
        if not elapsed_ok():
            break
        search(5, Fraction(1, 1), k, [])
        if best_solution is not None and len(best_solution) == k:
            break

    if best_solution is not None:
        return best_solution

    # Fallback: DP approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
