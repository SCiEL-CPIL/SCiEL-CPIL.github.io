import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target range to cover: 1..1000
    # We want to maximize L such that 1..L are covered.
    # The seed covers 1..(number of primes >= 13).
    # There are 168 primes <= 1000. Primes >= 13 are 168 - 5 = 163.
    # So seed score is roughly 163.
    # We aim to cover 1..1000.
    
    # 1. Generate all primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    
    # 2. Precompute residue classes for each prime
    # For each prime p, and for each possible residue r, 
    # we need to know which integers in 1..1000 are covered.
    # However, for greedy set cover, we usually iterate over sets.
    # Here, for a fixed prime p, choosing residue r covers {n in 1..1000 | n % p == r}.
    # We want to pick (p, r) pairs.
    
    # Budget: sum(1/p) <= 1
    # We will track used_primes and current_cost.
    
    covered = [False] * 1001  # 1..1000
    used_primes = set()
    current_cost = Fraction(0)
    selected_classes = []
    
    # To make greedy efficient, we can precompute for each prime p, 
    # the count of currently uncovered numbers for each residue r.
    # But this changes as we add classes.
    
    # Optimization: 
    # The "value" of adding (p, r) is the number of newly covered integers.
    # We want to maximize this value, subject to cost.
    # Since we have a hard budget of 1, and small primes have high cost,
    # we should be careful with small primes.
    
    # Heuristic: 
    # 1. Sort primes by size? No, large primes cover fewer numbers in 1..1000 per residue.
    #    Actually, a prime p covers approx 1000/p numbers per residue.
    #    Sum of 1/p for all primes up to 1000 is ~ 1.9. 
    #    We have budget 1. So we can afford roughly half the "coverage power" of all primes.
    #    Small primes cover more numbers per unit cost? 
    #    Cost of p is 1/p. Coverage per residue is ~1000/p.
    #    Ratio (Coverage/Cost) is ~1000. It's constant!
    #    So efficiency is similar for all primes in terms of density.
    #    However, small primes have fewer residues (p residues) but each residue class is denser?
    #    No, density is 1/p.
    #    
    #    The constraint is distinct primes.
    #    Let's use a standard Greedy Set Cover approach.
    #    At each step, find the (p, r) that covers the most *new* integers.
    #    Tie-breaker: lower cost (larger p) to save budget? Or higher coverage?
    #    Since we must stop when budget is exceeded or no more coverage, 
    #    we should prioritize high coverage.
    
    # To speed up, we can limit the search space.
    # We don't need to check all 168 primes * p residues every step if we are smart.
    # But 168 * 1000 operations is small. 168 steps max.
    # 168 * (168 * 1000) ~ 28 million ops. Python might be slow.
    # Let's optimize the inner loop.
    
    # Precompute for each prime p, the list of indices in 1..1000 for each residue.
    # residues[p][r] = list of n in 1..1000 s.t. n % p == r
    residues = {}
    for p in primes:
        res_list = [[] for _ in range(p)]
        for n in range(1, 1001):
            res_list[n % p].append(n)
        residues[p] = res_list
        
    # Greedy Loop
    while time.time() < deadline:
        best_gain = 0
        best_p = None
        best_r = None
        
        # Iterate through available primes
        # We can skip primes that are already used
        for p in primes:
            if p in used_primes:
                continue
            
            # Check if we can afford this prime
            # Cost is 1/p. 
            # We don't know which residue we'll pick, but cost is same for all r for a given p.
            # If current_cost + 1/p > 1, we cannot pick ANY class from p.
            if current_cost + Fraction(1, p) > 1:
                continue
                
            res_list = residues[p]
            
            # For each residue r, calculate gain
            # Gain = number of n in res_list[r] such that covered[n] is False
            # We can do this faster if we maintain a "uncovered" set or bitset.
            # Let's use a simple count for now, or a precomputed array of uncovered counts.
            
            # Optimization: Maintain an array `uncovered_count_for_residue`?
            # That's complex to update.
            
            # Let's just iterate.
            for r in range(p):
                if not res_list[r]:
                    continue
                
                # Count uncovered
                gain = 0
                for n in res_list[r]:
                    if not covered[n]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
        
        if best_p is None or best_gain == 0:
            break
            
        # Select best_p, best_r
        used_primes.add(best_p)
        current_cost += Fraction(1, best_p)
        selected_classes.append([best_p, best_r])
        
        # Update covered array
        for n in residues[best_p][best_r]:
            covered[n] = True
            
        # Check time again
        if time.time() > deadline:
            break
        
        # Optional: Early stop if 1..1000 is fully covered
        # We can check if covered[1..1000] are all True.
        # To avoid O(1000) check every step, we can track a counter `remaining_uncovered`.
        
    # Calculate final score L
    # Score is largest L such that 1..L are covered.
    L = 0
    for i in range(1, 1001):
        if covered[i]:
            L = i
        else:
            break
            
    return selected_classes
# EVOLVE-BLOCK-END
