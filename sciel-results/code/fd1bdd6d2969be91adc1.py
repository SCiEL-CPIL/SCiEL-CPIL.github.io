"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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
        
    def can_add(S, new_elem):
        """
        Check if adding new_elem to set S keeps it valid.
        S is a set of ints. new_elem is not in S.
        We need to check all triples (a, b, c) involving new_elem are valid.
        Let a, b be distinct elements in S.
        Triples: (new_elem, a, b), (a, new_elem, b), (a, b, new_elem).
        1. new_elem does not divide a * b
        2. a does not divide new_elem * b
        3. b does not divide new_elem * a (symmetric to 2)
        """
        S_list = list(S)
        m = len(S_list)
        # Check condition 1: new_elem | a*b
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if divides_product(new_elem, a, b):
                    return False
        
        # Check condition 2 & 3: a | new_elem*b or b | new_elem*a
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if i == j:
                    continue
                b = S_list[j]
                # Check a | new_elem * b
                if divides_product(a, new_elem, b):
                    return False
                # Note: b | new_elem * a is checked when i and j are swapped in the loop?
                # No, the loop covers all ordered pairs (i, j) with i != j.
                # So when i=idx(a), j=idx(b), we check a | x*b.
                # When i=idx(b), j=idx(a), we check b | x*a.
                # So both are covered.
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

    best_set = set()
    
    # Strategy: Multiple Randomized Greedy
    # We run several iterations of greedy construction with random orderings.
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_set = set()
        # Randomize order
        shuffled_candidates = candidates.copy()
        random.shuffle(shuffled_candidates)
        
        # Optional: Bias towards larger numbers or numbers with fewer factors?
        # Let's just use pure random for diversity, maybe weighted for some runs.
        
        for a in shuffled_candidates:
            if time.time() > deadline - 10:
                break
            if can_add(current_set, a):
                current_set.add(a)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Improvement on the best set found
    # Try to add any remaining element
    # If no element can be added, try swaps?
    # For now, just do a pass of adding if possible (should be none if greedy finished, 
    # but maybe different order in greedy left space? No, greedy checks all. 
    # However, a different order might allow different set. 
    # The local search here is to try to "fix" the set or expand it if we didn't check all? 
    # Actually, if greedy finished, no element from the *shuffled* list could be added. 
    # But we shuffled, so we checked all. So no element can be added.
    
    # To escape local optima, we can try removing one element and seeing if we can add two?
    # That's expensive. 
    # Let's try a simple swap: Remove u, Add v. If valid, and size is same, 
    # it doesn't help size. 
    # So we need to find a sequence of swaps that increases size? 
    # This is hard.
    
    # Alternative: Run more restarts if time permits.
    while time.time() < deadline - 5:
        # Try a few more restarts
        current_set = set()
        shuffled_candidates = candidates.copy()
        random.shuffle(shuffled_candidates)
        
        for a in shuffled_candidates:
            if time.time() > deadline - 5:
                break
            if can_add(current_set, a):
                current_set.add(a)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            # If we found a better one, we might want to try to extend it further?
            # It's already maximal for that order.
            
    # Final Verification
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
