import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    # We only care about primes that can fit in the budget and are useful.
    # Smallest prime is 2. 1/2 = 0.5.
    # We want to cover 1..L.
    
    # 2. Greedy Set Cover / Max Coverage approach
    # State: 
    # - covered: set of integers in 1..1000 that are covered
    # - budget: Fraction representing remaining budget (1 - sum(1/p))
    # - used_primes: set of primes already used
    
    max_L_target = 1000
    covered = set()
    budget = Fraction(1)
    used_primes = set()
    
    # Precompute coverage for each prime and residue?
    # That's 168 primes * avg 500 residues = 84k sets. Too much memory/time to precompute all.
    # Instead, we can evaluate candidates on the fly or use a heuristic.
    
    # A better approach for this specific problem (covering 1..L):
    # The "gap" is the first integer not covered.
    # We want to cover 1, then 2, then 3, etc.
    
    # Let's try a different greedy: 
    # At each step, find the prime p and residue r such that:
    # 1. p is not in used_primes
    # 2. 1/p <= budget
    # 3. The class (p, r) covers the smallest uncovered integer k.
    #    Actually, we want to maximize the extension of the contiguous prefix.
    
    # Let's track the current contiguous prefix length L.
    # L is the largest integer such that 1..L are all in `covered`.
    
    current_L = 0
    
    while time.time() < deadline:
        # Check if we have covered up to 1000
        if current_L >= 1000:
            break
            
        # Find the next uncovered integer
        next_uncovered = current_L + 1
        
        best_gain = -1
        best_p = None
        best_r = None
        
        # We need to find a prime p and residue r such that next_uncovered ≡ r (mod p)
        # And p is not used, and 1/p <= budget.
        # Among those, which one covers the most *additional* integers in the range [current_L+1, current_L+100] (or similar window)?
        # Or simply: which one covers the most new integers in 1..1000?
        
        # To speed up, we only consider primes p such that 1/p <= budget.
        # Since budget starts at 1, we can use all primes initially.
        # As we add primes, budget decreases.
        
        # Let's evaluate all unused primes.
        for p in primes:
            if p in used_primes:
                continue
            if Fraction(1, p) > budget:
                continue
            
            # The residue r must be such that next_uncovered % p == r
            r = next_uncovered % p
            
            # Calculate how many new integers this class covers in 1..1000
            # Class covers: r, r+p, r+2p, ... 
            # Note: if r == 0, it covers p, 2p, ... (since we want positive integers, 0 is not in 1..1000 unless we consider 0, but task says 1..L)
            # Task: "every integer 1..L lies in some class". 0 <= r < p.
            # If r=0, the numbers are p, 2p, ...
            # If r>0, the numbers are r, r+p, ...
            
            # We only care about numbers in 1..1000.
            # And we want to know how many of these are *not* in `covered`.
            
            # Optimization: We don't need to check all numbers if the prime is large.
            # If p > 1000, it's not in our list.
            # Max number of elements in class within 1..1000 is floor(1000/p).
            
            count_new = 0
            # Start from the smallest number in the class >= 1
            if r == 0:
                start = p
            else:
                start = r
            
            # Iterate through the class
            # We can break early if we find a gap in the contiguous prefix? No, we just count new covers.
            # But we want to prioritize classes that extend the prefix.
            # Actually, if it covers `next_uncovered`, it might cover `next_uncovered + p`, etc.
            
            # Let's just count how many new integers are covered in 1..1000.
            # This is O(1000/p).
            
            for n in range(start, 1001, p):
                if n not in covered:
                    count_new += 1
                    # Early termination? If count_new is already huge, maybe we can stop?
                    # No, we need exact count for comparison.
            
            if count_new > best_gain:
                best_gain = count_new
                best_p = p
                best_r = r
                
        if best_p is None:
            break
            
        # Commit the choice
        used_primes.add(best_p)
        budget -= Fraction(1, best_p)
        
        # Update covered set
        if best_r == 0:
            start = best_p
        else:
            start = best_r
            
        for n in range(start, 1001, best_p):
            covered.add(n)
            
        # Update current_L
        # current_L is the largest k such that 1..k are in covered.
        # Since we just added a class covering current_L+1, current_L might increase.
        while current_L < 1000 and (current_L + 1) in covered:
            current_L += 1
            
        # If we can't cover current_L + 1 anymore, we are stuck.
        # But the greedy loop will try to find a prime covering current_L + 1.
        # If no prime can cover it (because all are used or budget exhausted), it will break.
        
    # Construct the result
    classes = [[p, r] for p, r in zip([x[0] for x in used_primes], [x[1] for x in used_primes])] # This is wrong, used_primes is a set of ints
    
    # We need to store the pairs.
    # Let's refactor the loop to store (p, r) pairs.
    
    # Re-implementing with proper storage
    
    # --- Restart with proper storage ---
    covered = set()
    budget = Fraction(1)
    used_pairs = []
    current_L = 0
    primes_set = set(primes)
    
    while time.time() < deadline:
        if current_L >= 1000:
            break
            
        next_uncovered = current_L + 1
        
        best_gain = -1
        best_p = None
        best_r = None
        
        for p in primes:
            if any(pair[0] == p for pair in used_pairs):
                continue
            if Fraction(1, p) > budget:
                continue
            
            r = next_uncovered % p
            
            count_new = 0
            if r == 0:
                start = p
            else:
                start = r
                
            for n in range(start, 1001, p):
                if n not in covered:
                    count_new += 1
                    
            if count_new > best_gain:
                best_gain = count_new
                best_p = p
                best_r = r
                
        if best_p is None:
            break
            
        used_pairs.append((best_p, best_r))
        budget -= Fraction(1, best_p)
        
        if best_r == 0:
            start = best_p
        else:
            start = best_r
            
        for n in range(start, 1001, best_p):
            covered.add(n)
            
        while current_L < 1000 and (current_L + 1) in covered:
            current_L += 1
            
    return [[p, r] for p, r in used_pairs]
