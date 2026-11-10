"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # Also, for each number, store the pairs that form a bad triple with it
    # conflict_pairs[x] = list of (a, b) such that (a, b, x) is a bad triple
    # We need to be careful with ordering. Let's just store the set of triples first.
    triples_set = set()
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    triples_set.add((a, b, c))
                    
    if time.time() > deadline - 10:
        # Fallback to simple greedy if precomputation is too slow (unlikely for 128)
        selected = []
        for x in range(1, 129):
            if all(not (lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected

    # Build conflict map: for each x, which pairs (u, v) with u < v form a bad triple with x?
    # Actually, it's easier to store for each x, the list of pairs (u, v) such that {u, v, x} is bad.
    # We can optimize the check: 
    # If we have a set S, adding x is valid if there is no pair {u, v} subset of S such that {u, v, x} is bad.
    
    # Let's create a dictionary: conflict[x] = set of frozenset({u, v})
    conflict = {i: [] for i in range(1, n + 1)}
    for a, b, c in triples_set:
        # (a, b, c) is a bad triple
        # For x=c, the conflicting pair is (a, b)
        # For x=b, the conflicting pair is (a, c) -- wait, order matters for lookup?
        # Let's just store pairs as sorted tuples.
        pair_bc = (b, c)
        pair_ac = (a, c)
        pair_ab = (a, b)
        
        # If we pick c, we cannot have both a and b in S.
        conflict[c].append((a, b))
        # If we pick b, we cannot have both a and c in S.
        conflict[b].append((a, c))
        # If we pick a, we cannot have both b and c in S.
        conflict[a].append((b, c))

    # Heuristic Search
    best_set = []
    
    # Strategy 1: Greedy by minimum conflict count
    # Count how many bad triples each number is in
    conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
    
    # Try several orderings
    orderings = []
    
    # Ordering 1: Ascending conflict count
    orderings.append(sorted(numbers, key=lambda x: conflict_count[x]))
    
    # Ordering 2: Descending value (Primes tend to be higher? No, primes are scattered. But large numbers have fewer divisors? 
    # Actually, numbers with many divisors are more likely to be in bad triples.
    # Let's try sorting by number of divisors (ascending).
    def num_divisors(x):
        count = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
    div_counts = {i: num_divisors(i) for i in numbers}
    orderings.append(sorted(numbers, key=lambda x: div_counts[x]))
    
    # Ordering 3: Randomized restarts (if time permits)
    import random
    random.seed(42)
    for _ in range(50):
        orderings.append(sorted(numbers, key=lambda x: random.random()))
        
    # Also try the original order
    orderings.append(numbers)
    
    def greedy_solve(order):
        S = set()
        # To make the conflict check fast, we can maintain a set of "forbidden pairs" for the current S?
        # No, the check is: for x in order, check if any pair in conflict[x] is subset of S.
        # Since S grows, we can just check.
        # Optimization: Precompute for each x, the list of pairs.
        # When checking x, iterate through conflict[x]. If any pair (u,v) has u in S and v in S, reject x.
        
        for x in order:
            # Check if x can be added
            can_add = True
            for u, v in conflict[x]:
                if u in S and v in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S

    best_size = 0
    best_result = None
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_result = S
            
        # Local Search / Hill Climbing on the best found so far?
        # Let's try to improve the best_result found so far in subsequent iterations
        # Actually, let's just run all orderings and pick the best.
        
    if best_result is None:
        # Fallback
        best_result = {1}
    else:
        # One more pass: try to add any remaining numbers to the best set found
        # This handles cases where order mattered but a number could fit in the final set
        # but was skipped due to earlier conflicts that might have been resolved? 
        # No, greedy is monotonic. If it was rejected, it's because of existing elements.
        # But maybe removing a low-value element allows adding two high-value ones?
        # Let's do a simple swap optimization on the best set.
        
        current_set = list(best_result)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to swap each element in current_set with an element not in it
            # If the swap increases size or maintains size but allows more additions?
            # Simpler: Just try to add any element not in the set to a copy, 
            # and if it fits, keep it. (This is already done by greedy if the order allowed it, 
            # but maybe the order prevented it? No, greedy checks all candidates against the *current* set.
            # If an element was rejected, it was because of conflicts with elements that *were* added.
            # If we remove one of those elements, the rejected element might fit.
            # This is a 1-swap local search.
            
            for i, out_elem in enumerate(current_set):
                # Try removing out_elem
                trial_set = set(current_set)
                trial_set.remove(out_elem)
                
                # Try adding any element not in trial_set
                for in_elem in numbers:
                    if in_elem in trial_set:
                        continue
                    # Check if in_elem can be added to trial_set
                    can_add = True
                    for u, v in conflict[in_elem]:
                        if u in trial_set and v in trial_set:
                            can_add = False
                            break
                    if can_add:
                        # Perform swap
                        trial_set.add(in_elem)
                        # Check if this swap leads to a larger set? 
                        # It maintains size. But it might unlock future additions?
                        # To keep it simple and fast, let's only accept if it allows adding MORE elements immediately?
                        # Or just accept any valid swap and see if we can add more after?
                        # Let's just accept the swap and restart the "add if possible" loop for this new set.
                        # This is getting complex. Let's stick to the best greedy result.
                        # The greedy with multiple orderings is usually very strong for N=128.
                        pass
                        
    return sorted(list(best_result))

# EVOLVE-BLOCK-END
