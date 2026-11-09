"""
Strategy: Start with highly composite/smooth numbers to minimize product set, 
then use Simulated Annealing to balance sumset and product set sizes.
The key insight is that sets of smooth numbers have small product sets. 
We then tweak elements to reduce the sumset size without blowing up the product set.
"""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n = len(A)
        sumset = set()
        prodset = set()
        
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sumset.add(ai + aj)
                prodset.add(ai * aj)
        
        return max(len(sumset), len(prodset)), len(sumset), len(prodset)

    def fast_evaluate_delta(A, idx, new_val):
        """
        Approximate change in score when replacing A[idx] with new_val.
        This is a heuristic to speed up SA.
        """
        # In a full implementation, we'd track incremental changes.
        # For simplicity and robustness in 60s, we might just do full eval 
        # if SA steps are not too many, or use a cheap proxy.
        # Given n=20, full eval is O(n^2) = 400 ops. 
        # 10,000 steps * 400 ops = 4M ops, which is fine in Python if optimized?
        # Actually, set construction in Python is slow.
        # Let's stick to full evaluation but keep step count manageable.
        pass

    # 1. Generate initial candidates
    # Smooth numbers (5-smooth or 7-smooth) tend to have small product sets.
    # Let's generate a pool of candidates from 1..10000 that are "smooth"
    # or just start with a geometric progression and perturb.
    
    # Initial set: Geometric progression with ratio ~1.2 or 1.3?
    # Or start with small integers and scale?
    # Let's try a mix: Start with a set of highly composite numbers.
    
    # Pool of "good" candidates: numbers with many divisors or small prime factors.
    # Let's just use a random subset of 1..10000 but biased towards small numbers?
    # No, product set size is minimized by geometric progressions.
    # Sum set size is minimized by arithmetic progressions? No, AP has large sumset.
    # Actually, for fixed size, AP minimizes sumset? No, AP sumset is 2n-1.
    # GP sumset is large (exponential).
    # We want to balance.
    
    # Let's start with a geometric progression.
    # a, ar, ar^2, ...
    # If r is integer, product set is small.
    # Let's try r=2. 1, 2, 4, ..., 2^19 = 524288 > 10000.
    # So we can't use pure GP with r=2 for n=20.
    # We can use r=1.something? No, integers.
    
    # Alternative: Start with a set of small integers, e.g., 1..20.
    # |1..20 + 1..20| = 39.
    # |1..20 * 1..20| = ?
    # Products of 1..20. 1*1=1 ... 20*20=400.
    # Distinct products are many.
    # Let's check score for 1..20.
    # Sumset size: 39.
    # Product set size: Let's estimate.
    
    # Let's run a quick internal check on 1..20
    # But we need a better starting point.
    
    # Strategy:
    # 1. Generate several diverse initial sets.
    #    a) 1..20
    #    b) Geometric-like: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 nums). Need 6 more.
    #       Fill gaps with smooth numbers?
    #    c) Random smooth numbers.
    
    # Let's define a function to get k-smooth numbers up to M.
    def get_smooth_numbers(limit, max_prime=7, count=100):
        smooths = []
        for i in range(1, limit + 1):
            x = i
            p = 2
            while p * p <= x and p <= max_prime:
                while x % p == 0:
                    x //= p
                p += 1
            if x == 1 or (x > 1 and x <= max_prime):
                smooths.append(i)
            if len(smooths) >= count:
                break
        return smooths

    # We want 20 numbers.
    # Let's pick 20 random 7-smooth numbers from 1..10000?
    # Or better, construct a specific set.
    
    # Let's try a "Hybrid" set:
    # Small numbers to keep sums small, but spaced out to keep products distinct?
    # Actually, if numbers are close, sums collide (good for sumset size).
    # If numbers are multiplicatively related, products collide (good for prodset size).
    
    # Let's use Simulated Annealing starting from 1..20.
    
    best_set = list(range(1, 21))
    best_score, best_sum, best_prod = evaluate(best_set)
    
    current_set = list(range(1, 21))
    current_score, current_sum, current_prod = evaluate(current_set)
    
    temperature = 100.0
    cooling_rate = 0.95
    min_temp = 1.0
    
    # To speed up evaluation, we can precompute sumset and prodset structures?
    # No, just rely on n=20 being small.
    # 20*20/2 = 210 pairs.
    # 210 additions, 210 multiplications.
    # Set insertion is O(1) avg.
    # Total ~ 420 ops per eval.
    # 10,000 evals = 4.2M ops. Should be < 5 seconds in Python.
    # We have 75 seconds. We can do more.
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps and temperature > min_temp:
        steps += 1
        
        # Perturbation: Replace one element with a random value
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Suggest new value
        # Bias towards values that might reduce sumset (close to neighbors)
        # or prodset (multiplicative relations).
        # Simple: Random int 1..10000
        # Better: Local search around old_val
        # Let's try:
        # 1. Swap with another random element?
        # 2. Replace with old_val +/- delta?
        # 3. Replace with a random smooth number?
        
        # Let's try replacing with a random number in 1..10000
        # But to keep it efficient, maybe restrict range?
        # If we go too large, prodset explodes.
        # Let's restrict to 1..2000 initially, then expand?
        # Or just random 1..10000.
        
        new_val = random.randint(1, max_val)
        
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_set:
            continue
            
        # Perform replacement
        current_set[idx] = new_val
        
        new_score, new_sum, new_prod = evaluate(current_set)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            # Accept
            current_score = new_score
            current_sum = new_sum
            current_prod = new_prod
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Reject
            current_set[idx] = old_val
            
        temperature *= cooling_rate
        
        # If temperature is very low, we might get stuck.
        # Reheat occasionally?
        if steps % 1000 == 0 and temperature < 5.0:
            temperature = 50.0
            # Also randomize current set to escape local minima?
            # Let's just continue.
            
    # Final check
    if best_set is None:
        return list(range(1, 21))
        
    # Ensure distinct and positive
    # The SA loop ensures distinctness.
    
    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    n = 20
    max_val = 10000
    
    def evaluate(A):
        A = sorted(A)
        n = len(A)
        sumset = set()
        prodset = set()
        
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sumset.add(ai + aj)
                prodset.add(ai * aj)
        
        return max(len(sumset), len(prodset))

    # Initial set: 1..20
    best_set = list(range(1, 21))
    best_score = evaluate(best_set)
    
    current_set = list(range(1, 21))
    current_score = best_score
    
    temperature = 100.0
    cooling_rate = 0.99
    min_temp = 1.0
    
    steps = 0
    max_steps = 20000  # Adjust based on speed
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Perturbation strategy:
        # Try to move towards "smoother" numbers or close to existing ones
        # Let's pick a random neighbor or a random smooth number
        # For simplicity, random 1..10000
        
        new_val = random.randint(1, max_val)
        
        if new_val == old_val:
            continue
            
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        
        new_score = evaluate(current_set)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
            
        temperature *= cooling_rate
        
        # Reheat if stuck
        if steps % 500 == 0 and temperature < 10.0:
            temperature = 50.0
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
