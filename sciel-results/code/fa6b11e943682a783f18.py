"""Improved program: precompute forbidden triples, greedy + local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute: for each x, bad_pairs_for[x] = set of (a,b) with a<b, a!=x, b!=x
    # such that {x,a,b} is a forbidden triple
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) != ab:
                    continue
                if _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        for (a, b) in bad_pairs_for[x]:
            if a in current_set and b in current_set:
                return False
        return True
    
    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    # Multiple greedy strategies
    best_set = greedy(range(1, n + 1))
    
    s = greedy(range(n, 0, -1))
    if len(s) > len(best_set):
        best_set = s
    
    # Order by conflict degree
    degrees = [len(bad_pairs_for[i]) for i in range(n + 1)]
    order_deg = sorted(range(1, n + 1), key=lambda x: (degrees[x], x))
    s = greedy(order_deg)
    if len(s) > len(best_set):
        best_set = s
    
    order_deg_desc = sorted(range(1, n + 1), key=lambda x: (-degrees[x], x))
    s = greedy(order_deg_desc)
    if len(s) > len(best_set):
        best_set = s
    
    # Local search: add single elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, n + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                improved = True
                break
    
    # Local search: 1-for-2 swaps (remove 1, add 2)
    for _ in range(50):  # bounded iterations
        if time.time() > deadline - 5:
            break
        found = False
        for x in list(best_set):
            trial = best_set - {x}
            added_count = 0
            for y in range(1, n + 1):
                if y == x or y in trial:
                    continue
                if can_add(y, trial):
                    trial.add(y)
                    added_count += 1
                    if added_count >= 2:
                        break
            if len(trial) > len(best_set):
                best_set = trial
                found = True
                break
        if not found:
            break
    
    # Final add pass
    for x in range(1, n + 1):
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
    
    return list(best_set)
# EVOLVE-BLOCK-END
