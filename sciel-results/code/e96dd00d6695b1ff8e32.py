"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from fractions import Fraction
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Precompute prime factorizations for fast coprimality checks
    def factorize(n):
        """Return set of prime factors of n"""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute factor sets for all numbers 1..511
    factor_sets = [None] * 512
    for i in range(1, 512):
        factor_sets[i] = factorize(i)
    
    def are_coprime(a, b):
        """Check if a and b are coprime using precomputed factor sets"""
        if a == 1 or b == 1:
            return True
        # Check if they share any prime factor
        for p in factor_sets[a]:
            if p in factor_sets[b]:
                return False
        return True
    
    def is_coprime_to_all(a, chosen):
        """Check if a is coprime to all elements in chosen"""
        for b in chosen:
            if not are_coprime(a, b):
                return False
        return True
    
    def score_set(chosen):
        return sum(Fraction(1, 512 - a) for a in chosen)
    
    def greedy_with_order(order):
        chosen = []
        for a in order:
            if is_coprime_to_all(a, chosen):
                chosen.append(a)
        return chosen
    
    # Generate multiple orderings
    orderings = []
    
    # Ordering 1: Ascending a (maximizes individual contributions)
    orderings.append(list(range(1, 512)))
    
    # Ordering 2: Descending a
    orderings.append(list(range(511, 0, -1)))
    
    # Ordering 3: Prioritize numbers with fewer prime factors (more flexible)
    orderings.append(sorted(range(1, 512), key=lambda x: (len(factor_sets[x]), x)))
    
    # Ordering 4: Prioritize primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [a for a in range(2, 512) if is_prime(a)]
    composites = [a for a in range(1, 512) if not is_prime(a) and a != 1]
    orderings.append([1] + primes + composites)
    
    # Ordering 5: Sort by value descending, then by flexibility
    orderings.append(sorted(range(1, 512), key=lambda x: (-1.0/(512-x), len(factor_sets[x]))))
    
    # Try each ordering and keep the best
    best_set = None
    best_score = Fraction(0, 1)
    
    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break
        candidate = greedy_with_order(order)
        s = score_set(candidate)
        if s > best_score:
            best_score = s
            best_set = candidate
    
    # Local search with multiple strategies
    if best_set and time.time() < deadline:
        current_set = list(best_set)
        current_score = best_score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n = len(current_set)
            
            # Strategy 1: Try removing one element and adding a better one
            for i in range(n):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                # Find best replacement
                best_add = None
                best_add_score = current_score - Fraction(1, 512 - removed)
                
                for a in range(1, 512):
                    if a in remaining:
                        continue
                    if not is_coprime_to_all(a, remaining):
                        continue
                    new_score = current_score - Fraction(1, 512 - removed) + Fraction(1, 512 - a)
                    if new_score > best_add_score:
                        best_add_score = new_score
                        best_add = a
                
                if best_add is not None and best_add_score > current_score:
                    current_set[i] = best_add
                    current_score = best_add_score
                    improved = True
                    break
            
            # Strategy 2: Try adding a new element if we can
            if not improved and time.time() < deadline:
                for a in range(1, 512):
                    if a in current_set:
                        continue
                    if not is_coprime_to_all(a, current_set):
                        continue
                    new_score = current_score + Fraction(1, 512 - a)
                    if new_score > current_score:
                        current_set.append(a)
                        current_score = new_score
                        improved = True
                        break
            
            # Strategy 3: Try 2-swap moves (remove 2, add 2)
            if not improved and time.time() < deadline:
                n = len(current_set)
                found_swap = False
                for i in range(n):
                    if time.time() > deadline or found_swap:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline or found_swap:
                            break
                        removed1 = current_set[i]
                        removed2 = current_set[j]
                        remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                        
                        # Find best pair of replacements
                        best_pair = None
                        best_pair_score = current_score - Fraction(1, 512 - removed1) - Fraction(1, 512 - removed2)
                        
                        for a in range(1, 512):
                            if a in remaining or a == removed1 or a == removed2:
                                continue
                            if not is_coprime_to_all(a, remaining):
                                continue
                            for b in range(a+1, 512):
                                if b in remaining or b == removed1 or b == removed2:
                                    continue
                                if not is_coprime_to_all(b, remaining) or not are_coprime(a, b):
                                    continue
                                new_score = current_score - Fraction(1, 512 - removed1) - Fraction(1, 512 - removed2) + Fraction(1, 512 - a) + Fraction(1, 512 - b)
                                if new_score > best_pair_score:
                                    best_pair_score = new_score
                                    best_pair = (a, b)
                        
                        if best_pair is not None and best_pair_score > current_score:
                            current_set[i] = best_pair[0]
                            current_set[j] = best_pair[1]
                            current_score = best_pair_score
                            improved = True
                            found_swap = True
                            break
        
        best_set = current_set
        best_score = current_score
    
    # Randomized restarts if time permits
    if time.time() < deadline - 20:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Create a randomized ordering
            order = list(range(1, 512))
            random.shuffle(order)
            
            candidate = greedy_with_order(order)
            s = score_set(candidate)
            
            if s > best_score:
                best_score = s
                best_set = candidate
            
            # Quick local search on this candidate
            if time.time() < deadline - 10:
                current_set = list(candidate)
                current_score = s
                improved = True
                
                while improved and time.time() < deadline - 5:
                    improved = False
                    n = len(current_set)
                    
                    for i in range(n):
                        if time.time() > deadline - 5:
                            break
                        removed = current_set[i]
                        remaining = [x for j, x in enumerate(current_set) if j != i]
                        
                        best_add = None
                        best_add_score = current_score - Fraction(1, 512 - removed)
                        
                        for a in range(1, 512):
                            if a in remaining:
                                continue
                            if not is_coprime_to_all(a, remaining):
                                continue
                            new_score = current_score - Fraction(1, 512 - removed) + Fraction(1, 512 - a)
                            if new_score > best_add_score:
                                best_add_score = new_score
                                best_add = a
                        
                        if best_add is not None and best_add_score > current_score:
                            current_set[i] = best_add
                            current_score = best_add_score
                            improved = True
                            break
                    
                    if not improved and time.time() < deadline - 5:
                        for a in range(1, 512):
                            if a in current_set:
                                continue
                            if not is_coprime_to_all(a, current_set):
                                continue
                            new_score = current_score + Fraction(1, 512 - a)
                            if new_score > current_score:
                                current_set.append(a)
                                current_score = new_score
                                improved = True
                                break
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
    
    return best_set if best_set else list(range(511, 0, -1))
# EVOLVE-BLOCK-END
