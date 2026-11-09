"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit)
    
    N = 256
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    # Collect primes
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute prime factorization for all numbers 2..256
    factorization = [None] * (N + 1)
    for n in range(2, N + 1):
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1
        if temp > 1:
            factors.append((temp, 1))
        factorization[n] = factors
    
    def can_divide(a, b, c):
        """Check if a divides b*c using prime factorizations."""
        b_factors = dict(factorization[b])
        c_factors = dict(factorization[c])
        for p, exp_a in factorization[a]:
            exp_bc = b_factors.get(p, 0) + c_factors.get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
    
    def check_pair_conflict(a, b, c_set):
        """Check if adding a to c_set creates a conflict with b (where b is in c_set).
        We need: for all distinct x,y in c_set (x != y, both != a), a | x*y is false.
        This is equivalent to: for all x in c_set, for all y in c_set, x != y, a | x*y is false."""
        # This is expensive; instead check incrementally
        pass
    
    def is_valid_set(s):
        """Check if set s satisfies the constraint for all triples."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = s_list[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = s_list[k]
                    if can_divide(a, b, c):
                        return False
        return True
    
    # Start with all primes
    current_set = set(primes)
    best_set = set(primes)
    
    # Try to add composites one by one (greedy)
    composites = [i for i in range(2, N + 1) if not is_prime[i]]
    
    # Sort composites by complexity (fewer distinct prime factors first, then by value)
    def complexity(x):
        f = factorization[x]
        return (len(f), x)
    
    composites.sort(key=complexity)
    
    for c in composites:
        if time.time() > deadline - 10:
            break
        test_set = current_set | {c}
        if is_valid_set(test_set):
            current_set = test_set
    
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # Local search: try swapping elements
    all_numbers = list(range(2, N + 1))
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_list = sorted(best_set)
        for a in current_list:
            if time.time() > deadline - 5:
                break
            for b in all_numbers:
                if b in best_set:
                    continue
                # Try swapping a with b
                new_set = set(best_set)
                new_set.discard(a)
                new_set.add(b)
                if is_valid_set(new_set):
                    # Same size but different composition - accept if it might lead to improvement
                    # Actually, for pure swap we need a reason. Let's only accept if it enables future adds.
                    # For now, skip pure swaps and focus on add/remove strategies
                    pass
            if time.time() > deadline - 5:
                break
    
    # Try: remove one element, then greedily add back multiple
    for a in list(best_set):
        if time.time() > deadline - 5:
            break
        reduced_set = set(best_set)
        reduced_set.discard(a)
        # Try to add composites back
        temp_set = set(reduced_set)
        for c in composites:
            if c == a:
                continue
            if time.time() > deadline - 5:
                break
            test_set = temp_set | {c}
            if is_valid_set(test_set):
                temp_set = test_set
        if len(temp_set) > len(best_set):
            best_set = temp_set
    
    # Final verification
    if not is_valid_set(best_set):
        best_set = set(primes)
    
    result = sorted(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
