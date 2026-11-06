import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [p for p in range(2, 1001) if is_prime(p)]
    prime_indices = {p: i for i, p in enumerate(primes)}
    
    # 2. Initial Greedy Construction
    # We want to cover 1..L.
    # Strategy: Iteratively cover the first uncovered integer.
    # To keep cost low, we might prefer larger primes? 
    # However, small primes cover more integers in a given range [1, L].
    # Let's try a mix: Start with small primes, then optimize.
    
    # Let's initialize with the first 10 primes and assign residues to cover 1..100
    # Then use local search to expand.
    
    # Actually, a better initialization:
    # Use primes in ascending order. For each prime, assign residue to cover the next hole.
    
    classes = {} # p -> r
    current_L = 0
    covered = set()
    
    # Helper to check coverage up to L
    def get_covered_length(prime_residues, limit=5000):
        # prime_residues: dict p -> r
        if not prime_residues:
            return 0
        # Create a boolean array for 1..limit
        is_covered = [False] * (limit + 1)
        for p, r in prime_residues.items():
            # Mark multiples of p that are congruent to r
            # r can be 0, in which case it covers p, 2p, ...
            # If r == 0, first is p. If r > 0, first is r (if r < p).
            # Note: We cover 1..L.
            # Integers k such that k % p == r.
            # Smallest positive k:
            if r == 0:
                k = p
            else:
                k = r
            # Iterate k, k+p, k+2p...
            while k <= limit:
                is_covered[k] = True
                k += p
        
        # Find the largest L such that 1..L are all covered
        L = 0
        for i in range(1, limit + 1):
            if is_covered[i]:
                L = i
            else:
                break
        return L

    # Initialize with a greedy pass
    # We will use a list of active primes.
    active_primes = []
    
    # Let's try to build a solution that covers at least 100
    # Sort primes by size? 
    # Let's use the first few primes.
    for p in primes[:10]:
        # Find the first uncovered number
        # We need to know what is covered so far.
        # This is slow if done naively.
        pass
    
    # Faster Initialization:
    # Just pick a set of primes and random residues, then optimize.
    # Or, use the "next hole" greedy with a lookahead.
    
    # Let's use a simple greedy:
    # Maintain a set of covered numbers up to a limit (e.g., 2000).
    # For each step, find the smallest n not covered.
    # Pick a prime p not yet used.
    # Which p? 
    # If we pick small p, we cover many future numbers.
    # Let's pick the smallest available prime.
    
    covered_bool = [False] * 5000
    current_L = 0
    used_primes = set()
    
    # Pre-compute cost
    # We will track cost as Fraction
    
    # Greedy Step:
    # We want to maximize L.
    # While cost < 1:
    #   Find first hole n.
    #   If n > 1000, break (we are aiming for high L, but 1000 is the target).
    #   Pick a prime p.
    #   To be efficient, maybe pick p such that it covers n and many others.
    #   Small p covers more. Let's pick smallest available p.
    
    # However, this might use too much cost quickly.
    # Sum of reciprocals of first 10 primes is ~0.9.
    # Sum of first 11 primes (2,3,5,7,11,13,17,19,23,29,31) is ~0.94.
    # Sum of first 12 primes ... ~0.97.
    # Sum of first 13 primes ... ~0.99.
    # So we can use at most ~12-13 small primes.
    
    # Let's start with the first 12 primes.
    init_primes = primes[:12]
    for p in init_primes:
        # Assign residue to cover the first hole?
        # Or just random?
        # Let's assign residue to cover 1, then 2, etc.
        pass
        
    # Better Init:
    # Assign residues greedily to cover 1, 2, 3...
    covered_bool = [False] * 5000
    active_classes = {}
    
    for p in init_primes:
        # Find first uncovered n
        n = 1
        while n < 5000 and covered_bool[n]:
            n += 1
        if n >= 5000: break
        r = n % p
        active_classes[p] = r
        # Update covered_bool
        if r == 0:
            k = p
        else:
            k = r
        while k < 5000:
            covered_bool[k] = True
            k += p
            
    current_L = 0
    for i in range(1, 5000):
        if covered_bool[i]:
            current_L = i
        else:
            break
            
    best_L = current_L
    best_classes = dict(active_classes)
    best_cost = sum(Fraction(1, p) for p in active_classes)
    
    # Local Search / Random Kicks
    # We will try to improve L by changing residues or swapping primes.
    
    # Helper to update coverage efficiently?
    # For now, recalculate L from scratch if a change is accepted.
    # Optimization: Only recalculate if the change *could* improve L.
    
    # To speed up, we can maintain a "holes" set or just recalculate L up to current_L + delta.
    
    def calculate_L(classes_dict, limit=5000):
        if not classes_dict: return 0
        is_cov = [False] * (limit + 1)
        for p, r in classes_dict.items():
            if r == 0:
                k = p
            else:
                k = r
            while k <= limit:
                is_cov[k] = True
                k += p
        L = 0
        for i in range(1, limit + 1):
            if is_cov[i]:
                L = i
            else:
                break
        return L

    # Re-calculate initial L with the helper to be safe
    current_L = calculate_L(active_classes)
    best_L = current_L
    best_classes = dict(active_classes)
    
    # If we haven't reached 999, try to improve.
    
    # Available primes for swapping/adding
    all_primes = primes
    
    # Track cost
    current_cost = sum(Fraction(1, p) for p in active_classes)
    
    while time.time() - start_time < timeout:
        # 1. Try changing a residue
        if active_classes:
            p_to_change = random.choice(list(active_classes.keys()))
            old_r = active_classes[p_to_change]
            new_r = random.randint(0, p_to_change - 1)
            
            if new_r != old_r:
                active_classes[p_to_change] = new_r
                new_L = calculate_L(active_classes)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = dict(active_classes)
                # If new_L == best_L, we might keep it to explore further, 
                # but to avoid cycles, we might revert if it doesn't help.
                # Let's accept if it doesn't decrease L, to allow exploration.
                # But if it decreases L, revert.
                if new_L < current_L:
                    active_classes[p_to_change] = old_r
                else:
                    current_L = new_L
                    # If we improved best, update best
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = dict(active_classes)
            else:
                active_classes[p_to_change] = old_r # No change effectively
        
        # 2. Try swapping a prime
        # Pick a used prime and an unused prime
        used_list = list(active_classes.keys())
        if used_list:
            p_used = random.choice(used_list)
            # Find an unused prime
            unused_primes = [p for p in all_primes if p not in active_classes]
            if unused_primes:
                p_new = random.choice(unused_primes)
                
                # Check cost
                # Cost change: -1/p_used + 1/p_new
                # We need sum <= 1
                # Current cost is tracked? 
                # Let's just check if the new set is valid.
                
                # Remove p_used, add p_new
                r_old = active_classes.pop(p_used)
                
                # Assign residue to p_new. 
                # Strategy: Try to cover the first hole with p_new.
                # Find first hole
                # We need to know coverage without p_used.
                # This is expensive. 
                # Approximation: Just assign random residue or residue covering a small number.
                # Let's assign residue to cover 1.
                r_new = 1 % p_new
                active_classes[p_new] = r_new
                
                # Check cost
                new_cost = sum(Fraction(1, p) for p in active_classes)
                if new_cost > 1:
                    # Revert
                    active_classes[p_new] = r_old # Wait, r_old was for p_used.
                    # We removed p_used. We need to put it back.
                    active_classes[p_used] = r_old
                    del active_classes[p_new]
                else:
                    new_L = calculate_L(active_classes)
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = dict(active_classes)
                    if new_L < current_L:
                        # Revert
                        del active_classes[p_new]
                        active_classes[p_used] = r_old
                    else:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = dict(active_classes)
        
        # 3. Try adding a prime (if cost allows)
        unused_primes = [p for p in all_primes if p not in active_classes]
        if unused_primes:
            # Pick a small unused prime to be efficient?
            p_add = random.choice(unused_primes[:20]) # Smaller primes are better for coverage
            current_cost = sum(Fraction(1, p) for p in active_classes)
            if current_cost + Fraction(1, p_add) <= 1:
                # Assign residue to cover first hole
                # Find first hole in current coverage
                # We can use calculate_L logic to find the first False
                # But we need the array.
                # Let's just assign random residue for now, or residue of 1.
                r_add = random.randint(0, p_add - 1)
                active_classes[p_add] = r_add
                
                new_L = calculate_L(active_classes)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = dict(active_classes)
                if new_L < current_L:
                    del active_classes[p_add]
                else:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = dict(active_classes)
                        
        # 4. Try removing a prime
        if len(active_classes) > 1:
            p_rem = random.choice(list(active_classes.keys()))
            r_rem = active_classes.pop(p_rem)
            new_L = calculate_L(active_classes)
            if new_L == current_L:
                # We removed a redundant prime. Keep it removed to save cost?
                # Or maybe we want to keep it for future swaps?
                # If it doesn't hurt L, and saves cost, it's good.
                # But we might want to use that cost for a better prime.
                # So keep it removed.
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = dict(active_classes)
            else:
                # Revert
                active_classes[p_rem] = r_rem
                
        # Early stop if we reach the theoretical limit or a very high number
        if best_L >= 10000:
            break

    # Return the best classes found
    # Format: [[p, r], ...]
    result = [[p, r] for p, r in best_classes.items()]
    return result

# EVOLVE-BLOCK-END
