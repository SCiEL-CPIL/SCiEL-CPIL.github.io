"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations for all numbers
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    factorizations = {}
    for i in range(2, n + 1):
        factorizations[i] = prime_factorize(i)
    
    def divides_product(a, b, c):
        """Check if a divides b*c using factorizations"""
        for p, exp_a in factorizations[a].items():
            exp_bc = factorizations[b].get(p, 0) + factorizations[c].get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
    
    def is_valid_set(S):
        """Check if set S is valid: no three distinct a,b,c with a|b*c"""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(m):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True
    
    def can_add(S, new_elem):
        """Check if adding new_elem to set S keeps it valid"""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                # Check a | new_elem * b
                if divides_product(a, new_elem, b):
                    return False
                # Check new_elem | a * b
                if divides_product(new_elem, a, b):
                    return False
        return True
    
    # Precompute conflict counts: for each a, number of pairs (b,c) in candidates
    # with b != a, c != a, b != c such that a | b*c
    conflict_count = {}
    for a in candidates:
        cnt = 0
        for b in candidates:
            if b == a:
                continue
            for c in candidates:
                if c == a or c == b:
                    continue
                if divides_product(a, b, c):
                    cnt += 1
        conflict_count[a] = cnt
    
    if time.time() > deadline - 15:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Multiple greedy strategies
    best_set = []
    
    # Strategy 1: Sort by conflict count (ascending) - prefer less conflicting numbers
    sorted_by_conflict = sorted(candidates, key=lambda x: conflict_count[x])
    
    selected = set()
    for a in sorted_by_conflict:
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 2: Sort by number of prime factors (ascending) - primes first
    sorted_by_pf = sorted(candidates, key=lambda x: sum(factorizations[x].values()))
    
    selected = set()
    for a in sorted_by_pf:
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 3: Sort by value descending (large numbers are "safer")
    selected = set()
    for a in sorted(candidates, reverse=True):
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 4: Sort by value ascending
    selected = set()
    for a in sorted(candidates):
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Local improvement: try swapping elements to increase set size
    # Try removing one element and adding a different one
    best_set_set = set(best_set)
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for remove_elem in list(best_set_set):
            if time.time() > deadline - 5:
                break
            candidate_set = best_set_set - {remove_elem}
            for add_elem in candidates:
                if add_elem in candidate_set:
                    continue
                if can_add(candidate_set, add_elem):
                    # Try this swap
                    new_set = candidate_set | {add_elem}
                    if is_valid_set(new_set):
                        best_set_set = new_set
                        improved = True
                        break
    
    best_set = list(best_set_set)
    
    # Verify the result
    if not is_valid_set(best_set):
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Ensure all elements are Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
