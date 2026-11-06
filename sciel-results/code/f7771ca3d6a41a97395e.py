import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    LIMIT = 10**18
    
    # Generate a pool of primes. 
    # We know we can have at most 15 distinct prime factors.
    # The product of the first 15 is ~6e17.
    # We can swap in larger primes as long as product <= 10^18.
    # Let's generate primes up to 1000 to have a good pool for greedy selection.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_pool = sieve(1000)
    
    def count_coprime_pairs(divisors):
        """
        Count adjacent coprime pairs in a sorted list of divisors.
        """
        if len(divisors) < 2:
            return 0
        count = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                count += 1
            prev = curr
        return count

    def expand_divisors(divisors, p, e):
        """
        Efficiently expand the sorted divisor list by multiplying by p^0..p^e.
        Since divisors is sorted and p >= 2, the new divisors can be merged efficiently.
        However, for simplicity and correctness in a greedy step where we test many candidates,
        we can generate the new list and sort it. To speed up, we note that 
        d * p^k for fixed k is sorted. We are merging e+1 sorted lists.
        For small e (we expect e to be small for large p, larger for small p), 
        a simple sort of the combined list is O(N log N). 
        N is at most ~10^5 for 10^18, so sort is fast.
        """
        new_divs = []
        p_pow = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        new_divs.sort()
        return new_divs

    # State:
    # current_divisors: sorted list of divisors of current n
    # current_n: value of n
    # current_score: tau_perp(n)
    # used_primes: set of primes used
    
    current_divisors = [1]
    current_n = 1
    current_score = 0
    used_primes = set()
    
    # We will greedily add prime factors.
    # At each step, we try to add a new prime p with exponent e >= 1.
    # We want to maximize the increase in score.
    # To keep the search space manageable, we consider adding p^1, p^2, etc.
    # But adding p^1 is usually the first step.
    # Actually, we can treat "adding prime p" as a move. 
    # The exponent can be optimized locally after choosing the prime?
    # Or we can try (p, e) pairs.
    
    # A better greedy approach:
    # 1. Start with n=1, divisors=[1], score=0.
    # 2. Loop:
    #    - Find the best (p, e) such that n * p^e <= LIMIT and p not in used_primes.
    #    - "Best" means maximizing new_score - current_score.
    #    - If no such (p, e) improves the score (or no valid p exists), stop.
    #    - Update state.
    
    # To make this efficient, we should limit the range of p and e.
    # p should be in primes_pool.
    # e should be such that p^e <= LIMIT / current_n.
    # Also, e shouldn't be too large if p is small? 
    # Actually, for a fixed p, increasing e adds more divisors.
    # Does it always increase score? Not necessarily, it might cluster divisors.
    # But generally, more divisors = more chances for coprime pairs.
    # Let's test e=1, 2, ... up to max_e for each candidate p.
    
    # Optimization: 
    # Instead of testing all p in primes_pool (216 primes) and all e, 
    # we can be smarter.
    # However, 216 * 20 (max e) * (cost of expand+sort+score) 
    # Cost of expand+sort: N log N. N grows.
    # Initial N=1.
    # After 15 primes, N can be large.
    # If we do this greedily 15 times, the total cost is manageable.
    
    # Let's refine the "Best" selection.
    # We want to maximize delta_score.
    # If delta_score is 0, we might still want to add if it allows future better moves?
    # Greedy is myopic. But let's try pure greedy first.
    # If greedy gets stuck, we might need look-ahead or backtracking.
    # But the directive says "keep this design family" (search for max tau_perp).
    # Greedy is a valid search strategy.
    
    # To improve greedy, we can use a "soft" score or consider the "potential" of the prime.
    # But let's stick to exact delta_score.
    
    # One issue: Adding a large prime p=100 with e=1 vs p=2 with e=5.
    # p=2^5 adds 5 divisors (2,4,8,16,32) to the existing set.
    # p=100 adds 1 divisor (100).
    # The structure of divisors matters.
    
    # Let's implement the greedy loop.
    
    while time.time() < DEADLINE:
        best_delta = -1
        best_p = None
        best_e = None
        best_new_divs = None
        best_new_n = None
        best_new_score = None
        
        # Iterate through candidate primes
        # We can restrict p to primes > current max prime? No, we can use any unused prime.
        # But using small primes allows higher exponents.
        # Using large primes keeps n small but adds few divisors.
        
        # Optimization: Sort primes_pool by some heuristic? 
        # Or just iterate all.
        # To speed up, we can break early if p^1 > LIMIT/current_n.
        
        for p in primes_pool:
            if p in used_primes:
                continue
            
            max_e = 1
            while (current_n * (p ** (max_e + 1))) <= LIMIT:
                max_e += 1
            # Cap max_e to avoid excessive computation if p is very small
            # e.g. p=2, max_e could be 60. 2^60 is huge.
            # But N (num divisors) would be 2^60? No, N is product of (e_i+1).
            # If we add 2^60, N becomes 61 * (previous N).
            # If previous N is 1000, new N is 61000. Sort is fine.
            # But 2^60 is ~1e18. 
            # Let's cap max_e to 20 for now? 
            # If p=2, 2^20 = 1e6. 
            # If we need to reach 1e18, we need more primes.
            # Actually, if we use high exponents of small primes, we use up the budget fast.
            # Greedy might pick 2^60 and stop. That's bad.
            # We should balance.
            
            # Heuristic: Limit max_e to keep the "cost" (log p * e) reasonable?
            # Or just try e=1, 2, ... up to max_e but stop if delta starts decreasing?
            # For a fixed p, does delta_score always increase with e?
            # Not necessarily. 
            # Let's try e from 1 to max_e.
            
            # To optimize, we can pre-compute p_powers.
            p_pow = 1
            for e in range(1, max_e + 1):
                p_pow *= p
                new_n = current_n * p_pow
                if new_n > LIMIT:
                    break
                
                # Calculate new divisors
                new_divs = expand_divisors(current_divisors, p, e)
                new_score = count_coprime_pairs(new_divs)
                delta = new_score - current_score
                
                if delta > best_delta:
                    best_delta = delta
                    best_p = p
                    best_e = e
                    best_new_divs = new_divs
                    best_new_n = new_n
                    best_new_score = new_score
            
            # Early termination? 
            # If p is large, max_e is 1.
            # If p is small, we check many e.
            
        if best_p is None:
            break
            
        # If best_delta <= 0, we might still want to add if it's 0?
        # Adding a divisor that doesn't increase score but might enable future increases?
        # In a strict greedy, we stop if no improvement.
        # However, sometimes 0 improvement is a bridge.
        # Let's require delta > 0 for now to avoid infinite loops or useless additions.
        # Actually, if delta == 0, n increases, reducing budget. Likely bad.
        
        if best_delta <= 0:
            break
            
        # Update state
        current_divisors = best_new_divs
        current_n = best_new_n
        current_score = best_new_score
        used_primes.add(best_p)
        
        # If we have used many primes, or n is close to limit, we might stop.
        # The loop condition handles time.
        
    return current_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
