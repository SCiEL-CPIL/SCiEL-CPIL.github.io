import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples and conflict structure
    conflict_count = [0] * (N + 1)
    pair_conflicts = {}
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
    
    def can_add(v, selected_list):
        """Check if v can be added using precomputed conflicts."""
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
    
    def greedy(order):
        """Greedy: add numbers in given order if no bad triple is formed."""
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
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
    
    # Strategy 9: Numbers with many prime factors
    order9 = sorted(range(1, N + 1), key=lambda x: (-total_omega(x), x))
    candidates.append(greedy(order9))
    
    # Strategy 10: Even numbers
    evens = [x for x in range(2, N + 1) if x % 2 == 0]
    candidates.append(greedy(evens))
    
    # Strategy 11: Multiples of 3
    mult3 = [x for x in range(3, N + 1) if x % 3 == 0]
    candidates.append(greedy(mult3))
    
    # Random restarts
    random.seed(42)
    num_random = 300
    for trial in range(num_random):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    # Weighted random with noise on conflict count
    for trial in range(100):
        if time.time() > deadline - 20:
            break
        order_w = sorted(range(1, N + 1), key=lambda x: (conflict_count[x] + random.random() * 10, x))
        candidates.append(greedy(order_w))
    
    # Pick best candidate
    best = max(candidates, key=len)
    best_size = len(best)
    
    # Local search: kick-and-refill strategy (faster than deterministic swap)
    def kick_and_refill(sel, deadline):
        """Randomized kick-and-refill: remove k random elements, refill greedily."""
        sel_list = list(sel)
        n = len(sel_list)
        
        def refill(trial_list):
            """Greedily add elements that can be added."""
            trial_set = set(trial_list)
            for v in range(1, N + 1):
                if v in trial_set:
                    continue
                if can_add(v, trial_list):
                    trial_list.append(v)
                    trial_set.add(v)
            return trial_list
        
        improved = True
        kick_sizes = [1, 2, 3]
        while improved and time.time() < deadline:
            improved = False
            for kick_size in kick_sizes:
                if time.time() > deadline:
                    break
                # Try multiple kicks
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    if n < kick_size:
                        continue
                    # Remove kick_size random elements
                    remove_indices = random.sample(range(n), min(kick_size, n))
                    remove_set = set(sel_list[i] for i in remove_indices)
                    trial = [x for x in sel_list if x not in remove_set]
                    # Refill
                    trial = refill(trial)
                    if len(trial) > n:
                        sel_list = trial
                        n = len(sel_list)
                        improved = True
                        break
        return sel_list
    
    # Also try deterministic add
    def local_search_add(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel_list:
                    continue
                if can_add(v, sel_list):
                    sel_list.append(v)
                    improved = True
        return sel_list
    
    if time.time() < deadline:
        best = local_search_add(best, deadline)
    
    if time.time() < deadline - 5:
        best = kick_and_refill(best, deadline)
    
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
