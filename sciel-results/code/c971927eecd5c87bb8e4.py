"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = gen_primes(1000)
    MAX_L = 100000
    
    def get_prefix(covered):
        L = 0
        while L + 1 <= MAX_L and covered[L + 1]:
            L += 1
        return L

    def calculate_coverage(classes, MAX_L):
        covered = [False] * (MAX_L + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for n in range(start, MAX_L + 1, p):
                covered[n] = True
        return covered

    # 1. Greedy Initialization
    # We use a slightly different greedy: prioritize small primes to maximize density, 
    # but pick the residue that covers the most numbers in the range [1, 1000] 
    # while keeping the prefix as long as possible.
    
    def greedy_init():
        covered = [False] * (MAX_L + 1)
        used_primes = set()
        classes = []
        budget = Fraction(1, 1)
        
        # Sort primes by size (smaller primes cover more numbers, cheaper budget per number)
        # But we must ensure distinct primes.
        
        # Heuristic: Try to cover the first uncovered number with the smallest possible prime
        # that hasn't been used, choosing the residue that also covers other nearby gaps.
        
        current_prefix = 0
        
        # Limit iterations to avoid infinite loops if stuck
        max_iter = 50
        
        for _ in range(max_iter):
            if current_prefix >= MAX_L:
                break
            
            first_uncovered = current_prefix + 1
            if first_uncovered > MAX_L:
                break
                
            best_gain = -1
            best_prime = None
            best_residue = None
            
            # Only consider primes that are not used
            # To speed up, we can limit to primes < 100 for the first few steps
            # but let's try all available
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget - cost < 0:
                    continue
                
                # The residue MUST cover first_uncovered
                r = first_uncovered % p
                
                # Calculate gain: how many new numbers are covered in [1, current_prefix + 1000]
                # We only need to check up to a certain horizon to estimate gain
                horizon = min(MAX_L, current_prefix + 1000)
                gain = 0
                # Check coverage in the window
                for n in range(1, horizon + 1):
                    if not covered[n]:
                        if n % p == r:
                            gain += 1
                            
                # Tie-breaker: prefer smaller primes (cheaper) or larger gain
                if gain > best_gain:
                    best_gain = gain
                    best_prime = p
                    best_residue = r
                elif gain == best_gain and best_prime is not None:
                    # Prefer smaller prime
                    if p < best_prime:
                        best_prime = p
                        best_residue = r

            if best_prime is None or best_gain <= 0:
                break
                
            # Apply
            p = best_prime
            r = best_residue
            used_primes.add(p)
            classes.append([p, r])
            budget -= Fraction(1, p)
            
            # Update coverage
            start = r if r > 0 else p
            for n in range(start, MAX_L + 1, p):
                covered[n] = True
                
            new_prefix = get_prefix(covered)
            if new_prefix <= current_prefix:
                # If prefix didn't grow, check if we should stop
                # Sometimes adding a class covers gaps but doesn't extend prefix yet
                # But if we can't extend prefix, we might be stuck.
                # Let's allow a few "non-extending" moves if they cover many gaps
                if best_gain < 10:
                    break
            current_prefix = new_prefix
            
        return classes, covered

    classes, covered = greedy_init()
    current_L = get_prefix(covered)
    
    # 2. Local Search Improvement
    # Try to improve the prefix length by modifying the classes.
    
    def try_improve(classes, covered, current_L):
        improved = True
        iterations = 0
        max_iterations = 1000 # Safety cap
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1. Try removing a class
            for i in range(len(classes)):
                if len(classes) <= 1:
                    continue
                # Temporarily remove
                classes.remove(classes[i])
                new_covered = calculate_coverage(classes, MAX_L)
                new_L = get_prefix(new_covered)
                
                if new_L > current_L:
                    current_L = new_L
                    covered = new_covered
                    improved = True
                else:
                    classes.insert(i, classes[i]) # Restore
            
            # 2. Try changing residue of an existing prime
            for i in range(len(classes)):
                p, r = classes[i]
                # Try a few random residues or specific ones
                # We want to cover the first uncovered number if possible
                first_uncovered = current_L + 1
                if first_uncovered > MAX_L:
                    break
                    
                # Candidate residues: one that covers first_uncovered
                target_r = first_uncovered % p
                if target_r == r:
                    continue # Already covers it
                
                # Check if changing r improves prefix
                old_r = r
                classes[i] = [p, target_r]
                new_covered = calculate_coverage(classes, MAX_L)
                new_L = get_prefix(new_covered)
                
                if new_L > current_L:
                    current_L = new_L
                    covered = new_covered
                    improved = True
                else:
                    classes[i] = [p, old_r]
            
            # 3. Try swapping a prime for a different one
            # This is expensive. We'll skip for now or do limited search.
            
            # 4. Try adding a new class (if budget allows)
            # Check budget
            total_cost = sum(Fraction(1, p) for p, r in classes)
            if total_cost < 1:
                # Find first uncovered
                first_uncovered = current_L + 1
                if first_uncovered <= MAX_L:
                    # Find best new prime
                    best_gain = -1
                    best_p = None
                    best_r = None
                    used_primes = {p for p, r in classes}
                    
                    for p in primes:
                        if p in used_primes:
                            continue
                        cost = Fraction(1, p)
                        if total_cost + cost > 1:
                            continue
                        
                        r = first_uncovered % p
                        # Estimate gain in horizon
                        horizon = min(MAX_L, current_L + 500)
                        gain = 0
                        for n in range(1, horizon + 1):
                            if not covered[n] and n % p == r:
                                gain += 1
                                
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                            best_r = r
                            
                    if best_p is not None and best_gain > 0:
                        classes.append([best_p, best_r])
                        new_covered = calculate_coverage(classes, MAX_L)
                        new_L = get_prefix(new_covered)
                        if new_L > current_L:
                            current_L = new_L
                            covered = new_covered
                            improved = True
                        else:
                            classes.pop()

        return classes, covered, current_L

    classes, covered, final_L = try_improve(classes, covered, current_L)
    
    # Final verification
    total = sum(Fraction(1, p) for p, r in classes)
    if total > 1:
        # Fallback: remove classes until valid
        # Sort by impact? Just remove largest primes (smallest cost) first? No, remove smallest primes (largest cost) to free budget?
        # Actually, if sum > 1, we violated constraint. 
        # The greedy and local search should have respected it, but let's be safe.
        # We will return the best valid subset.
        # For now, if invalid, we return empty or a known valid small set.
        pass
        
    return classes
# EVOLVE-BLOCK-END
