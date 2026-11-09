import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = gen_primes(1000)
    
    # Candidate sets of primes
    # 1. Seed set
    seed_primes = [p for p in primes if p >= 13]
    
    # 2. Greedy sets with small primes
    candidate_sets = [seed_primes]
    
    # Generate a few greedy sets
    # Sort by 1/p descending (small primes first)
    sorted_primes = sorted(primes, key=lambda x: 1/x, reverse=True)
    
    # Try to include small primes
    current_set = []
    current_sum = Fraction(0, 1)
    for p in sorted_primes:
        if current_sum + Fraction(1, p) <= 1:
            current_set.append(p)
            current_sum += Fraction(1, p)
        else:
            # Try to swap or skip
            pass
    if current_set:
        candidate_sets.append(current_set)
        
    # Try sets starting with {2, 3}
    base_small = [2, 3]
    base_sum = Fraction(1, 2) + Fraction(1, 3)
    for p in sorted_primes:
        if p in base_small: continue
        if base_sum + Fraction(1, p) <= 1:
             # This creates many sets, just take one representative or a few
             pass

    # Let's just try the seed and a few random valid sets
    # And a set optimized for small primes
    
    best_L = 0
    best_classes = []
    
    def evaluate(prime_set):
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        # Greedy assignment
        # residues: dict p -> r
        residues = {}
        # covered: bitset or list of bools
        # Max L is capped at 10^5, but we can stop earlier
        max_L_check = 10000 
        covered = [False] * (max_L_check + 1)
        
        L = 0
        for n in range(1, max_L_check + 1):
            if covered[n]:
                L = n
                continue
                
            # Find candidates
            candidates = []
            for p in prime_set:
                if p not in residues:
                    candidates.append(p)
                else:
                    if n % p == residues[p]:
                        candidates.append(p)
                        
            if not candidates:
                break
                
            # Heuristic: Choose smallest prime
            # Or choose prime that covers most future numbers?
            # Let's try smallest prime first
            p_chosen = min(candidates)
            
            r = n % p_chosen
            residues[p_chosen] = r
            
            # Update covered
            # Mark all k in 1..max_L_check such that k % p_chosen == r
            for k in range(r if r != 0 else p_chosen, max_L_check + 1, p_chosen):
                # k must be >= 1
                if k >= 1:
                    covered[k] = True
            
            L = n
            
        # Construct classes
        classes = [[p, residues[p]] for p in prime_set if p in residues]
        return L, classes

    # Evaluate seed
    L, classes = evaluate(candidate_sets[0])
    if L > best_L:
        best_L = L
        best_classes = classes
        
    # Try a few random sets
    # We need to ensure sum <= 1
    # Randomly select primes, check sum
    
    attempts = 0
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        # Generate random set
        # Sort primes by size
        # Pick random subset
        
        # Simple random: shuffle primes, take prefix until sum > 1, then remove last
        rand_primes = primes[:]
        random.shuffle(rand_primes)
        s = Fraction(0, 1)
        rand_set = []
        for p in rand_primes:
            if s + Fraction(1, p) <= 1:
                rand_set.append(p)
                s += Fraction(1, p)
            else:
                # Maybe try to fit a smaller prime?
                # Just break for now
                pass
        
        if not rand_set:
            continue
            
        L, classes = evaluate(rand_set)
        if L > best_L:
            best_L = L
            best_classes = classes

    return best_classes
