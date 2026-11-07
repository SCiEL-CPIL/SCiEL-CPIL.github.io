"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # All primes up to 40
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_prime_dims = len(all_primes)
    
    # Factorize each number 2..40 into exponent vector
    def factorize(n):
        vec = [0] * num_prime_dims
        for i, p in enumerate(all_primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    candidates = list(range(2, 41))
    vecs = {n: factorize(n) for n in candidates}
    
    # Fast validity check using exponent vectors
    def get_subset_sums(vec_list):
        """Compute all subset sum vectors for a list of exponent vectors."""
        sums = {tuple(0 for _ in range(num_prime_dims))}
        for v in vec_list:
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_prime_dims))
                if ns in sums or ns in new_sums:
                    return None  # Collision detected
                new_sums.add(ns)
            sums = sums | new_sums
        return sums
    
    def try_add_with_sums(sums, vec):
        """Try to add vec to a set with current subset sums. Returns new sums or None."""
        if sums is None:
            return None
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_prime_dims))
            if ns in sums or ns in new_sums:
                return None
            new_sums.add(ns)
        return sums | new_sums
    
    def is_valid(S):
        """Check if set S has distinct subset products."""
        vec_list = [vecs[n] for n in S]
        return get_subset_sums(vec_list) is not None
    
    def greedy_complete(S, deadline):
        """Greedily add elements to S until stuck. Returns (S, sums)."""
        S = set(S)
        vec_list = [vecs[n] for n in S]
        sums = get_subset_sums(vec_list)
        if sums is None:
            return S, None
        
        added = True
        while added and time.time() < deadline:
            added = False
            candidates_remaining = [c for c in candidates if c not in S]
            random.shuffle(candidates_remaining)
            for x in candidates_remaining:
                new_sums = try_add_with_sums(sums, vecs[x])
                if new_sums is not None:
                    S.add(x)
                    sums = new_sums
                    added = True
                    break
        return S, sums
    
    # Primes as initial set
    primes = [p for p in candidates if isqrt(p) < 2 or all(p % d for d in range(2, isqrt(p) + 1))]
    
    best_set = set(primes)
    best_size = len(best_set)
    best_sums = get_subset_sums([vecs[n] for n in best_set])
    
    # Randomized local search
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        # Choose starting point
        r = restart % 4
        if r == 0:
            # Start from primes
            S = set(primes)
        elif r == 1:
            # Start from random subset of primes + try adding composites
            S = set(random.sample(primes, max(1, len(primes) - 2)))
        elif r == 2:
            # Start from best_set with perturbation
            S = set(best_set)
            for _ in range(random.randint(1, 2)):
                if S:
                    S.discard(random.choice(list(S)))
        else:
            # Start from best_set, remove 1-3 elements
            S = set(best_set)
            for _ in range(random.randint(1, 3)):
                if S:
                    S.discard(random.choice(list(S)))
        
        # Greedy completion
        S, sums = greedy_complete(S, deadline - 1)
        
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
            best_sums = sums
            if best_size == 18:
                break
        
        # Try swap moves if not at max
        if len(S) < 18 and time.time() < deadline - 3:
            for swap_iter in range(30):
                if time.time() > deadline - 2:
                    break
                
                # 1-for-1 swap
                to_remove = random.choice(list(S))
                S.discard(to_remove)
                candidates_remaining = [c for c in candidates if c not in S]
                random.shuffle(candidates_remaining)
                improved = False
                for to_add in candidates_remaining:
                    # Recompute sums without to_remove
                    vec_list = [vecs[n] for n in S]
                    sums = get_subset_sums(vec_list)
                    if sums is None:
                        S.add(to_remove)
                        break
                    new_sums = try_add_with_sums(sums, vecs[to_add])
                    if new_sums is not None:
                        S.add(to_add)
                        sums = new_sums
                        improved = True
                        # Greedy complete after swap
                        S, sums = greedy_complete(S, deadline - 1)
                        if len(S) > best_size:
                            best_set = set(S)
                            best_size = len(S)
                            best_sums = sums
                            if best_size == 18:
                                break
                        break
                if not improved:
                    S.add(to_remove)
                
                if best_size == 18:
                    break
            
            if best_size == 18:
                break
    
    # Final verification using direct product check
    def verify_direct(S):
        products = {1}
        for n in S:
            new_products = set()
            for p in products:
                np = p * n
                if np in products or np in new_products:
                    return False
                new_products.add(np)
            products = products | new_products
        return True
    
    if not verify_direct(best_set):
        # Fallback to primes
        return [int(p) for p in primes]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
