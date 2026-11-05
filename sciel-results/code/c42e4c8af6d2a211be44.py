import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Sieve for primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    # We will use a subset of these primes.
    # Sort primes in increasing order to try small ones first? 
    # Or just keep them as is.
    
    # We want to cover 1..L. Let's set a high limit for coverage check.
    # The task says capped at 10^5, but we likely won't reach that with budget 1.
    # Let's aim for L up to 5000 to be safe and fast.
    L_MAX = 5000
    covered = [False] * (L_MAX + 1)
    
    # Track used primes
    used_primes = set()
    
    # Budget tracking using Fractions for exactness
    current_sum = Fraction(0, 1)
    max_budget = Fraction(1, 1)
    
    solution = []
    
    # Greedy loop
    while time.time() < deadline:
        # Find smallest uncovered integer
        next_uncovered = -1
        for i in range(1, L_MAX + 1):
            if not covered[i]:
                next_uncovered = i
                break
        
        if next_uncovered == -1:
            # All covered up to L_MAX
            break
            
        # Find best prime to cover next_uncovered
        best_p = -1
        best_gain = -1
        best_r = -1
        
        # Try all unused primes
        for p in primes:
            if p in used_primes:
                continue
            
            # Check if adding [p, r] is within budget
            new_sum = current_sum + Fraction(1, p)
            if new_sum > max_budget:
                continue
            
            r = next_uncovered % p
            
            # Calculate gain: how many new integers in 1..L_MAX are covered?
            # We can approximate by checking a window or just counting.
            # Since L_MAX is 5000, and primes are up to 1000,
            # each class covers ~5000/p numbers.
            # Summing over all primes is expensive.
            # Let's just pick the prime that minimizes p (smallest prime) 
            # among those that fit in budget?
            # Smallest p has highest coverage density.
            if p < best_p or best_p == -1:
                best_p = p
                best_r = r
                best_gain = 1 # Placeholder, we are just picking smallest p for now
        
        if best_p == -1:
            # No prime fits in budget or all used
            break
            
        # Add the chosen prime
        used_primes.add(best_p)
        solution.append([best_p, best_r])
        current_sum += Fraction(1, best_p)
        
        # Update covered array
        # Numbers covered are k such that k % best_p == best_r
        for k in range(best_r, L_MAX + 1, best_p):
            if k >= 1: # Ensure we don't cover 0 if best_r is 0, but task is 1..L
                if k > 0:
                    covered[k] = True
            else:
                # If best_r is 0, the first positive is best_p
                # The loop range(best_r, ...) starts at 0.
                # We need to handle k=0 case or start from best_p.
                pass
        
        # Corrected update loop
        # If r == 0, covered numbers are p, 2p, ...
        # If r > 0, covered numbers are r, r+p, ...
        # My loop `range(best_r, L_MAX + 1, best_p)` includes 0 if best_r=0.
        # I should skip 0.
        
    # Calculate final L
    L = 0
    for i in range(1, L_MAX + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return solution
