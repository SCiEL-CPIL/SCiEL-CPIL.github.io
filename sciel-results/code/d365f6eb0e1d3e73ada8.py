"""Structural change: simulated annealing / local search over prime residue class assignments."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _coverage_length(classes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by the given classes.
    classes: list of [p, r] pairs.
    Returns L (the last consecutive covered integer starting from 1).
    """
    if not classes:
        return 0
    # Build a coverage array up to max_check
    # For efficiency, use a bytearray
    cov = bytearray(max_check + 1)
    for p, r in classes:
        if r == 0:
            start = p
        else:
            start = r
        if start > max_check:
            continue
        for n in range(start, max_check + 1, p):
            cov[n] = 1
    # Find the first gap
    L = 0
    for n in range(1, max_check + 1):
        if cov[n]:
            L = n
        else:
            break
    return L

def _total_cost(classes):
    """Compute sum of 1/p for all classes using exact rationals."""
    total = Fraction(0)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    primes = _sieve(1000)
    # We need sum(1/p) <= 1. Let's figure out which primes we can afford.
    # The sum of 1/p for all primes up to 1000 is about 2.198, so we can't use all.
    # We need to select a subset with sum <= 1.
    
    # Strategy: 
    # 1. Start with a good initial solution (greedy or seed)
    # 2. Use simulated annealing to optimize the assignment of residues
    # 3. Also try adding/removing primes to improve coverage
    
    # Initial solution: use the seed approach (primes >= 13 with r = i+1 mod p)
    # But let's also try a greedy approach to get a better starting point.
    
    # First, let's determine a set of primes we can afford.
    # Greedy: pick primes in order of 1/p (i.e., smallest p first) until budget is used.
    # But actually, for coverage, we want primes that are "useful" - not necessarily the smallest.
    # However, smaller primes cover more residues per unit cost, so they're generally better.
    
    # Let's try multiple strategies and keep the best.
    
    best_L = 0
    best_classes = []
    
    # Strategy 1: Seed solution
    primes_seed = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    seed_cost = _total_cost(seed_classes)
    if seed_cost <= Fraction(1):
        L = _coverage_length(seed_classes, 20000)
        if L > best_L:
            best_L = L
            best_classes = seed_classes[:]
    
    # Strategy 2: Greedy set cover
    # Pick primes one at a time, choosing the one that covers the most new numbers.
    # We'll work with a target range of about 5000 to keep it tractable.
    
    def greedy_initial(target_L=5000):
        """Greedy set cover to find initial prime-residue assignment."""
        used = {}
        cost = Fraction(0)
        covered = [False] * (target_L + 1)
        
        # Precompute which numbers each (p, r) would cover
        # For each prime p, the best residue r is the one that covers the most uncovered numbers.
        
        for _ in range(len(primes)):
            if time.time() > deadline:
                break
            if all(covered[n] for n in range(1, target_L + 1)):
                break
            
            best_gain = -1
            best_p = None
            best_r = None
            
            for p in primes:
                if p in used:
                    continue
                if cost + Fraction(1, p) > Fraction(1):
                    continue
                
                # Find best residue for this prime
                best_r_for_p = -1
                best_count_for_p = -1
                for r in range(p):
                    count = 0
                    if r == 0:
                        start = p
                    else:
                        start = r
                    if start > target_L:
                        continue
                    for n in range(start, target_L + 1, p):
                        if not covered[n]:
                            count += 1
                    if count > best_count_for_p:
                        best_count_for_p = count
                        best_r_for_p = r
                
                if best_count_for_p > best_gain:
                    best_gain = best_count_for_p
                    best_p = p
                    best_r = best_r_for_p
            
            if best_p is None or best_gain <= 0:
                break
            
            used[best_p] = best_r
            cost += Fraction(1, best_p)
            # Update covered
            p = best_p
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, target_L + 1, p):
                covered[n] = True
        
        return [[p, r] for p, r in used.items()]
    
    greedy_classes = greedy_initial(5000)
    if greedy_classes:
        L = _coverage_length(greedy_classes, 20000)
        if L > best_L:
            best_L = L
            best_classes = greedy_classes[:]
    
    # Now use simulated annealing to improve the solution
    # State: list of [p, r] pairs
    # Moves: 
    #   1. Change residue of an existing prime
    #   2. Replace a prime with another unused prime
    #   3. Add a new prime (if budget allows)
    #   4. Remove a prime (to free budget for another)
    
    if best_classes and time.time() < deadline:
        current = best_classes[:]
        current_L = best_L
        current_cost = _total_cost(current)
        
        # Precompute: for each prime in current, its index
        prime_set = set(p for p, r in current)
        
        # Temperature schedule
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        rng = random.Random(42)
        
        while time.time() < deadline and T > T_min:
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.5 and current:
                # Change residue of a random prime
                idx = rng.randint(0, len(current) - 1)
                p, old_r = current[idx]
                new_r = rng.randint(0, p - 1)
                if new_r == old_r:
                    T *= cooling
                    continue
                current[idx] = [p, new_r]
                
                # Evaluate
                L_new = _coverage_length(current, 20000)
                
                if L_new >= current_L:
                    # Accept
                    current_L = L_new
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = current[:]
                else:
                    # Reject (we only want to improve or maintain)
                    current[idx] = [p, old_r]
                
            elif move_type < 0.7 and current:
                # Replace a prime with another unused prime
                idx = rng.randint(0, len(current) - 1)
                p_old, r_old = current[idx]
                
                # Find unused primes
                unused_primes = [p for p in primes if p not in prime_set]
                if not unused_primes:
                    T *= cooling
                    continue
                
                # We can only replace if the new prime's cost doesn't exceed budget
                # Or if we remove p_old and add p_new
                p_new = rng.choice(unused_primes)
                
                # Check if replacing is feasible
                new_cost = current_cost - Fraction(1, p_old) + Fraction(1, p_new)
                if new_cost > Fraction(1):
                    T *= cooling
                    continue
                
                # Choose a random residue for the new prime
                r_new = rng.randint(0, p_new - 1)
                
                # Tentatively replace
                current[idx] = [p_new, r_new]
                prime_set.remove(p_old)
                prime_set.add(p_new)
                current_cost = new_cost
                
                L_new = _coverage_length(current, 20000)
                
                if L_new >= current_L:
                    current_L = L_new
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = current[:]
                else:
                    # Revert
                    current[idx] = [p_old, r_old]
                    prime_set.remove(p_new)
                    prime_set.add(p_old)
                    current_cost = _total_cost(current)
                
            elif move_type < 0.85 and len(current) < len(primes):
                # Add a new prime
                unused_primes = [p for p in primes if p not in prime_set]
                if not unused_primes:
                    T *= cooling
                    continue
                
                # Pick the smallest unused prime (cheapest)
                candidates = [p for p in unused_primes if current_cost + Fraction(1, p) <= Fraction(1)]
                if not candidates:
                    T *= cooling
                    continue
                
                p_new = rng.choice(candidates)
                r_new = rng.randint(0, p_new - 1)
                
                current.append([p_new, r_new])
                prime_set.add(p_new)
                current_cost += Fraction(1, p_new)
                
                L_new = _coverage_length(current, 20000)
                
                if L_new >= current_L:
                    current_L = L_new
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = current[:]
                else:
                    # Revert
                    current.pop()
                    prime_set.remove(p_new)
                    current_cost -= Fraction(1, p_new)
                
            else:
                # Remove a prime
                if len(current) <= 1:
                    T *= cooling
                    continue
                
                idx = rng.randint(0, len(current) - 1)
                p_old, r_old = current[idx]
                
                current.pop(idx)
                prime_set.remove(p_old)
                current_cost -= Fraction(1, p_old)
                
                L_new = _coverage_length(current, 20000)
                
                if L_new >= current_L:
                    current_L = L_new
                    if L_new > best_L:
                        best_L = L_new
                        best_classes = current[:]
                else:
                    # Revert
                    current.insert(idx, [p_old, r_old])
                    prime_set.add(p_old)
                    current_cost += Fraction(1, p_old)
            
            T *= cooling
        
        # After SA, do a final local search pass: for each prime, try all residues
        if time.time() < deadline:
            current = best_classes[:]
            current_L = best_L
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in range(len(current)):
                    p, old_r = current[idx]
                    best_r_local = old_r
                    best_L_local = current_L
                    for new_r in range(p):
                        if new_r == old_r:
                            continue
                        current[idx] = [p, new_r]
                        L_new = _coverage_length(current, 20000)
                        if L_new > best_L_local:
                            best_L_local = L_new
                            best_r_local = new_r
                    current[idx] = [p, best_r_local]
                    if best_L_local > current_L:
                        current_L = best_L_local
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = current[:]
                        improved = True
    
    # Final validation
    final_cost = _total_cost(best_classes)
    if final_cost > Fraction(1):
        # Remove classes until cost is OK
        best_classes.sort(key=lambda x: -1.0 / x[0])
        while final_cost > Fraction(1) and best_classes:
            p, r = best_classes.pop()
            final_cost -= Fraction(1, p)
    
    # Ensure distinct primes
    seen = set()
    filtered = []
    for p, r in best_classes:
        if p not in seen:
            seen.add(p)
            filtered.append([p, r])
    best_classes = filtered
    
    return best_classes

# EVOLVE-BLOCK-END
