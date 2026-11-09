import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    # Filter out very small primes? No, small primes are efficient.
    # But we have a budget of 1. Sum(1/p) for all primes <= 1000 is ~1.9.
    # We need to select a subset S of primes such that sum(1/p) <= 1.
    
    # Strategy:
    # 1. Select a candidate set of primes. 
    #    Since sum(1/p) <= 1, and sum of reciprocals of all primes is ~1.9,
    #    we can't use all. 
    #    Smaller primes are "expensive" but cover more numbers per residue class (density 1/p).
    #    Actually, density is 1/p for all. 
    #    However, small primes have fewer residue classes (p choices).
    #    Large primes have many residue classes (p choices).
    #    
    #    Let's try a mixed approach:
    #    - Include primes up to some limit P_small to ensure good density coverage.
    #    - Use remaining budget for larger primes to fill gaps.
    
    # Precompute coverage sets for all primes and residues?
    # 1000 numbers. 168 primes.
    # For each prime p, residue r covers {n in 1..1000 | n % p == r}.
    # Size of each set is approx 1000/p.
    
    # Let's use a randomized hill-climbing / simulated annealing approach.
    # State: A list of (p, r) pairs.
    # Objective: Maximize L such that 1..L are covered.
    # Constraint: sum(1/p) <= 1.
    
    # To make search tractable:
    # 1. Fix the set of primes to use? Or allow changing primes?
    #    Changing primes is hard. Let's first select a promising set of primes.
    #    Sort primes by 1/p (cost). 
    #    Greedy selection of primes by cost-efficiency?
    #    Efficiency of a prime p in covering 1..1000:
    #    It provides p different "patterns". 
    #    
    #    Let's try this:
    #    Phase 1: Select primes. 
    #    We want a set of primes P such that sum(1/p) <= 1.
    #    To maximize coverage, we want primes that are "independent" in their coverage.
    #    Small primes (2, 3, 5, 7) are very powerful because their residue classes are large.
    #    Let's always include small primes if budget allows?
    #    Cost(2)=0.5, Cost(3)=0.33, Sum=0.83. 
    #    Cost(5)=0.2. 0.83+0.2 > 1. 
    #    So we can have {2,3} (cost 0.83) + some larger primes.
    #    Or {2,5,7,11,13...}
    #    
    #    Let's evaluate a few candidate sets of primes manually or via simple heuristic.
    #    Actually, let's just run a local search on the full space of (p,r) assignments,
    #    but we need to handle the prime selection too.
    
    # Alternative Strategy:
    # Use the fact that L is capped at 1000.
    # We want to cover 1..1000.
    # 
    # Let's use a Greedy Set Cover with a twist:
    # 1. Start with empty cover.
    # 2. At each step, pick the (p,r) that maximizes "newly covered numbers" / "cost".
    #    Cost is 1/p.
    #    Value is count of new numbers.
    #    This is the standard greedy for weighted set cover (approximation).
    #    The previous program did this but was slow due to Python loops.
    #    I will optimize the inner loop using bitsets or numpy if possible, 
    #    or just efficient list operations.
    
    # Let's optimize the Greedy Set Cover.
    # Target: 1..1000.
    # Represent covered numbers as a bitset (integer or bytearray).
    # For each prime p, and residue r, precompute the bitset of numbers covered.
    
    # Bitset representation:
    # An integer where bit i is 1 if number (i+1) is covered?
    # 1000 bits. Python ints are arbitrary precision, so this is fast.
    
    # Precomputation:
    # masks[p][r] = bitset of n in 1..1000 such that n % p == r.
    
    masks = {}
    for p in primes:
        masks[p] = [0] * p
        for r in range(p):
            mask = 0
            # n ranges from 1 to 1000
            # First n >= 1 such that n % p == r
            # If r == 0, n = p, 2p, ...
            # If r > 0, n = r, r+p, ... (if r < p)
            # Note: n must be in 1..1000.
            
            # Calculate start
            if r == 0:
                start = p
            else:
                start = r
                if start < 1: start = 1 # r is 1..p-1 so start>=1 always
                # Wait, if r > 1000? No, r < p <= 1000.
                # But if r > 1000, no numbers. But p<=1000 so r<1000.
                # However, if r=1, start=1.
            
            # Iterate
            n = start
            while n <= 1000:
                mask |= (1 << (n - 1))  # Bit 0 corresponds to number 1
                n += p
            masks[p][r] = mask
    
    # Now, Greedy Set Cover.
    # current_cover = 0 (bitset)
    # used_primes = set()
    # current_cost = Fraction(0)
    # selected = []
    
    # To speed up "find best (p,r)", we need to evaluate gain for all available (p,r).
    # Gain = popcount(masks[p][r] & ~current_cover)
    # Cost = 1/p
    # Ratio = Gain / Cost = Gain * p
    
    # We want to maximize Ratio, subject to cost constraint.
    # If current_cost + 1/p > 1, skip p.
    
    # Optimization:
    # Instead of checking all 168 primes * avg 500 residues every step,
    # we can maintain a priority queue or just scan.
    # 168 steps max.
    # Scan cost: 168 primes * (p residues) * (popcount cost).
    # Popcount in Python int is fast (C implementation).
    # Total residues to check: Sum(p for p in primes) ~ 168 * 500 = 84,000.
    # 168 steps * 84,000 ops = 14 million popcounts.
    # Python might handle 14M simple ops in ~1-2 seconds? 
    # Bitwise AND and popcount are very fast.
    
    current_cover = 0
    used_primes = set()
    current_cost = Fraction(0)
    selected_classes = []
    
    # To avoid re-calculating gain for primes that are already "full" or similar,
    # we just re-scan.
    
    # List of primes sorted by size? No, just iterate.
    # We can pre-compute list of (p, r, mask) for all valid (p,r) to avoid dict lookup overhead?
    # Or just iterate primes and residues.
    
    # Let's try to run the greedy.
    
    # Optimization: Pre-compute list of all candidate (p, r, mask, cost)
    # But cost is 1/p.
    # We can filter candidates dynamically based on budget.
    
    # To make it faster, we can group by prime.
    # For each prime p, we find the best residue r that maximizes gain.
    # Then we compare across primes.
    
    while time.time() < deadline:
        best_gain_ratio = -1
        best_p = None
        best_r = None
        best_mask = None
        best_gain = 0
        
        # Iterate through primes
        for p in primes:
            if p in used_primes:
                continue
            
            # Check budget
            # If current_cost + 1/p > 1, we can't use this prime.
            # Using Fraction is slow. 
            # Let's use floats for budget check in the loop, and verify with Fraction at the end?
            # Risk: Floating point errors. 
            # Sum of reciprocals is around 1. 
            # 1/p is small. 
            # Let's use a float budget for speed, and keep track of exact cost.
            # If float_cost > 1.0, skip.
            
            # Actually, let's just use Fraction but it might be the bottleneck.
            # Alternative: Multiply all costs by LCM? No, too big.
            # Let's use a float `float_cost` for pruning and `exact_cost` for final check.
            # If float_cost + 1.0/p > 1.0 + 1e-9, skip.
            
            # Let's assume the previous program's slowness was due to Python loops counting elements.
            # Bitset popcount is O(1) effectively (word size).
            
            # Check if affordable (approx)
            if current_cost + Fraction(1, p) > 1:
                continue
            
            # Find best residue for this prime
            prime_best_gain = 0
            prime_best_r = -1
            prime_best_mask = 0
            
            # Iterate residues
            # Optimization: Only check residues that have non-zero mask?
            # All residues have non-zero mask for p <= 1000 and range 1..1000?
            # For p=997, r=1 covers 1, 998. r=2 covers 2, 999... 
            # Yes, all residues have at least one number if p <= 1000.
            
            for r in range(p):
                m = masks[p][r]
                if m == 0: continue
                # Calculate new bits
                new_bits = m & ~current_cover
                if new_bits == 0: continue
                gain = new_bits.bit_count()
                
                if gain > prime_best_gain:
                    prime_best_gain = gain
                    prime_best_r = r
                    prime_best_mask = m
            
            if prime_best_gain == 0:
                continue
                
            # Compare with global best
            # Ratio = gain * p (since cost is 1/p)
            ratio = prime_best_gain * p
            
            if ratio > best_gain_ratio:
                best_gain_ratio = ratio
                best_p = p
                best_r = prime_best_r
                best_mask = prime_best_mask
                best_gain = prime_best_gain
        
        if best_p is None or best_gain == 0:
            break
            
        # Select best
        used_primes.add(best_p)
        current_cost += Fraction(1, best_p)
        selected_classes.append([best_p, best_r])
        
        # Update cover
        current_cover |= best_mask
        
        # Check if we covered 1..1000?
        # Full mask for 1..1000
        full_mask = (1 << 1000) - 1
        if (current_cover & full_mask) == full_mask:
            break
            
    # Calculate L
    # Find largest L such that bits 0..L-1 are set in current_cover
    # Iterate from 1000 down to 1
    L = 0
    # Check if 1..1000 covered
    # We can do this by checking bit_count of (current_cover & ((1<<L)-1)) == L
    # But we want the *prefix* L.
    
    # Efficient way:
    # If bit i is 0, L <= i.
    # We can find the first 0 bit from the left (MSB side of the 1000 bits).
    
    # Let's just loop. 1000 iterations is negligible.
    for i in range(1, 1001):
        if not (current_cover & (1 << (i - 1))):
            break
        L = i
        
    return selected_classes

# EVOLVE-BLOCK-END
