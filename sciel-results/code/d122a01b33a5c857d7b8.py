import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples and conflict structure
    # pair_conflicts[(a,b)] = set of c values that form a bad triple with a,b
    # Also track for each element v, which pairs (a,b) would conflict with v
    conflict_count = [0] * (N + 1)
    pair_conflicts = {}
    # For fast can_add: for each v, store the set of pairs that conflict with v
    # i.e., elem_conflicts[v] = set of frozenset({a,b}) where (a,b,v) is bad
    elem_conflicts = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1
                    pair_conflicts.setdefault((a, b), set()).add(c)
                    pair_conflicts.setdefault((a, c), set()).add(b)
                    pair_conflicts.setdefault((b, c), set()).add(a)
                    elem_conflicts[a].add((b, c))
                    elem_conflicts[b].add((a, c))
                    elem_conflicts[c].add((a, b))
    
    if time.time() > deadline:
        return [1]
    
    def can_add_fast(v, selected_set, selected_list):
        """Check if v can be added using precomputed conflicts. O(len(selected))."""
        for i in range(len(selected_list)):
            a = selected_list[i]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                key = (a, b) if a < b else (b, a)
                if v in pair_conflicts.get(key, ()):
                    return False
        return True
    
    def can_add_optimized(v, selected_set):
        """Faster check: for each pair in selected, check if v conflicts.
        Use the fact that if (a,b) is in selected and v in pair_conflicts[(a,b)], conflict.
        We iterate over pairs in selected."""
        sel = list(selected_set)
        n = len(sel)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sel[i], sel[j]
                key = (a, b) if a < b else (b, a)
                if v in pair_conflicts.get(key, ()):
                    return False
        return True
    
    def can_add_v2(v, selected_list):
        """Most optimized: precompute which pairs in selected conflict with v."""
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            for j in range(i + 1, n):
                b = selected_list[j]
                key = (a, b) if a < b else (b, a)
                pc = pair_conflicts.get(key)
                if pc is not None and v in pc:
                    return False
        return True
    
    def greedy(order, can_add_fn=can_add_v2):
        """Greedy: add numbers in given order if no bad triple is formed."""
        selected = []
        selected_set = set()
        for x in order:
            if can_add_fn(x, selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Multiple greedy strategies
    candidates = []
    
    # Strategy 1: Natural order
    candidates.append(greedy(list(range(1, N + 1))))
    
    # Strategy 2: Reverse order
    candidates.append(greedy(list(range(N, 0, -1))))
    
    # Strategy 3: Sorted by conflict count (ascending) - least constrained first
    order3 = sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x))
    candidates.append(greedy(order3))
    
    # Strategy 4: Sorted by conflict count (descending) - most constrained first
    order4 = sorted(range(1, N + 1), key=lambda x: (-conflict_count[x], x))
    candidates.append(greedy(order4))
    
    # Strategy 5: Primes and 1
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    primes = [1] + [p for p in range(2, N + 1) if is_prime(p)]
    candidates.append(greedy(primes))
    
    # Strategy 6: Powers of 2
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128]
    candidates.append(greedy(powers2))
    
    # Strategy 7: Odd numbers
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    candidates.append(greedy(odds))
    
    # Strategy 8: Numbers with few prime factors
    def total_omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order8 = sorted(range(1, N + 1), key=lambda x: (total_omega(x), x))
    candidates.append(greedy(order8))
    
    # Strategy 9: Numbers with many prime factors (opposite of 8)
    order9 = sorted(range(1, N + 1), key=lambda x: (-total_omega(x), x))
    candidates.append(greedy(order9))
    
    # Strategy 10: Even numbers
    evens = [x for x in range(2, N + 1) if x % 2 == 0]
    candidates.append(greedy(evens))
    
    # Strategy 11: Multiples of 3
    mult3 = [x for x in range(3, N + 1) if x % 3 == 0]
    candidates.append(greedy(mult3))
    
    # Random restarts with different seeds - more of them since can_add is fast
    random.seed(42)
    num_random = 200
    for trial in range(num_random):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    # Additional random with weighted ordering by conflict count
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        # Weighted random: less-conflicted elements more likely to come first
        weights = [1.0 / (conflict_count[x] + 1) for x in range(1, N + 1)]
        order = list(range(1, N + 1))
        # Shuffle with bias toward low conflict
        for i in range(len(order) - 1, 0, -1):
            j = random.randint(0, i)
            order[i], order[j] = order[j], order[i]
        # Sort partially by conflict count with noise
        order11 = sorted(range(1, N + 1), key=lambda x: (conflict_count[x] + random.random() * 5, x))
        candidates.append(greedy(order11))
    
    # Pick best candidate
    best = max(candidates, key=len)
    best_size = len(best)
    
    # Local search: try to add elements
    def local_search_add(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel_list:
                    continue
                if can_add_v2(v, sel_list):
                    sel_list.append(v)
                    improved = True
        return sel_list
    
    # Local search: try swaps (remove one, add more)
    def local_search_swap(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_idx in range(len(sel_list)):
                if time.time() > deadline:
                    break
                removed = sel_list[remove_idx]
                trial = [x for x in sel_list if x != removed]
                added = 0
                for v in range(1, N + 1):
                    if v in trial:
                        continue
                    if can_add_v2(v, trial):
                        trial.append(v)
                        added += 1
                if added > 1:
                    sel_list = trial
                    improved = True
                    break
        return sel_list
    
    # 2-opt local search: remove 2 elements, try to add 3+
    def local_search_2opt(sel, deadline):
        sel_list = list(sel)
        n = len(sel_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    r1, r2 = sel_list[i], sel_list[j]
                    trial = [x for x in sel_list if x != r1 and x != r2]
                    added = 0
                    for v in range(1, N + 1):
                        if v in trial:
                            continue
                        if can_add_v2(v, trial):
                            trial.append(v)
                            added += 1
                    if added > 2:
                        sel_list = trial
                        n = len(sel_list)
                        improved = True
                        break
                if improved:
                    break
        return sel_list
    
    if time.time() < deadline:
        best = local_search_add(best, deadline)
    
    if time.time() < deadline:
        best = local_search_swap(best, deadline)
    
    if time.time() < deadline - 5:
        best = local_search_2opt(best, deadline)
    
    # Final validation
    def is_valid(sel):
        s = set(sel)
        if len(s) != len(sel) or len(s) == 0:
            return False
        for a, b, c in combinations(sel, 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True
    
    if not is_valid(best):
        return [1]
    
    if not best:
        return [1]
    
    return best
