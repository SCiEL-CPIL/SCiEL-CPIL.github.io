import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    # Primes to consider. Product of first 15 primes is ~6.1e17.
    # Including 16th prime (67) makes product ~4e19 > 1e18, so exponents must be 0 for larger primes mostly.
    # We will consider primes up to 47 (15 primes).
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    k = len(primes)
    LIMIT = 10**18

    # Precompute max exponents for each prime such that p^e <= LIMIT
    # This helps bound the search space slightly, though the product constraint is tighter.
    max_exp = [0] * k
    for i, p in enumerate(primes):
        e = 1
        val = p
        while val * p <= LIMIT:
            val *= p
            e += 1
        max_exp[i] = e

    def calc_score_from_exponents(exps):
        """
        Calculate tau_perp(n) for n = prod p_i^e_i.
        exps: list of integers.
        Returns (score, n). If n > LIMIT, returns (-1, n).
        """
        n = 1
        divisors = [1]
        
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            # Multiply n by p^e
            n *= (p ** e)
            if n > LIMIT:
                return -1, n
            
            # Update divisors
            # Current divisors are for prod_{j<i} p_j^{e_j}
            # New divisors are d * p^j for j=0..e
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            
            # Optimization: If the list of divisors gets too large, we might want to stop?
            # Max divisors for n <= 1e18 is around 103680. 
            # 100k divisors is manageable for sorting and gcd in Python if done efficiently.
            # However, sorting 100k items takes ~0.01s. 
            # We call this function many times. 
            # We need to be careful with time.
            
            divisors = new_divs
        
        if n < 2:
            return 0, n
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score, n

    # Initial state: Try the primorial of the first 15 primes
    # 2*3*5*...*47
    best_exps = [1] * k
    # Check if it fits
    n_check = 1
    for p in primes:
        n_check *= p
    if n_check > LIMIT:
        # Remove largest primes until it fits
        while n_check > LIMIT and k > 1:
            best_exps[-1] = 0
            n_check //= primes[-1]
            k -= 1 # Effectively reduce k, but we keep the list size for indexing consistency?
            # Actually, let's just zero out exponents for primes that don't fit in the primorial.
            # It's easier to just start with a valid state.
    
    # Let's find a valid starting point properly
    start_exps = [0] * k
    n_current = 1
    for i in range(k):
        p = primes[i]
        if n_current * p <= LIMIT:
            start_exps[i] = 1
            n_current *= p
        else:
            break
            
    best_exps = start_exps
    best_score, best_n = calc_score_from_exponents(best_exps)
    
    # Simulated Annealing
    # Temperature
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # We will perform steps until time runs out
    # Each step: pick a random prime index, change exponent by +/- 1 or random small amount
    # If n > LIMIT, reject (or handle penalty)
    
    # To speed up, we can precompute powers of primes
    prime_powers = []
    for i, p in enumerate(primes):
        powers = [1]
        val = 1
        for _ in range(max_exp[i] + 1):
            val *= p
            powers.append(val)
        prime_powers.append(powers)
        
    # Helper to get n quickly from exps? 
    # Recalculating n from scratch is O(k) which is 15. That's fast.
    # Recalculating divisors is the bottleneck.
    
    # Optimization: 
    # If we change exponent of p_i, can we update divisors incrementally?
    # Divisors of n = A * p^e.
    # Divisors are { d * p^j | d | A, 0 <= j <= e }.
    # If we change e to e', the set of divisors changes.
    # It is complex to update the sorted list and gcd counts incrementally without rebuilding.
    # Given 15 primes, max divisors ~100k. 
    # Rebuilding takes ~0.02s.
    # 120s / 0.02s = 6000 steps.
    # 6000 steps of simulated annealing might be enough to find a good local optimum.
    # Let's try to make it faster.
    
    # Alternative: Only evaluate if the change is likely to improve? No, SA needs to accept worse.
    
    # Let's use a faster divisor generation and gcd check.
    # numpy might help for gcd? No, math.gcd is C-optimized.
    
    # Let's try to optimize the inner loop of calc_score_from_exponents.
    # We can avoid sorting if we generate them in a structured way? 
    # No, divisors are not generated in sorted order naturally from the product construction.
    # However, we can use the property that if we maintain the list of divisors of A (sorted),
    # and multiply by p^j, the union of these arithmetic progressions can be merged?
    # That's complex. Sorting is probably fine for 100k items.
    
    # Let's refine the SA loop to be as fast as possible.
    
    current_exps = best_exps[:]
    current_score = best_score
    current_n = best_n
    
    # Random seed for reproducibility
    random.seed(42)
    
    while time.time() < DEADLINE:
        # Pick a random prime index
        i = random.randint(0, k - 1)
        
        # Pick a new exponent
        # Bias towards small changes
        delta = random.choice([-1, 1])
        if random.random() < 0.1:
            delta = random.choice([-2, 2])
            
        new_exp = current_exps[i] + delta
        
        if new_exp < 0 or new_exp > max_exp[i]:
            continue
            
        # Check if n will exceed limit
        # n_new = current_n / p^current_exp * p^new_exp
        # To avoid floating point issues, use integer math
        # We can check: current_n // (p^current_exp) * (p^new_exp) <= LIMIT
        # But we need to ensure current_n is divisible by p^current_exp (it is).
        
        p = primes[i]
        
        # Calculate new n
        # Optimization: if new_exp < current_exp, we are dividing.
        # if new_exp > current_exp, we are multiplying.
        
        # Let's just calculate n_new directly to be safe, it's fast.
        n_new = current_n
        if new_exp < current_exp:
            n_new //= (p ** (current_exp - new_exp))
        else:
            n_new *= (p ** (new_exp - current_exp))
            
        if n_new > LIMIT:
            continue
            
        # Calculate score
        # We need to update exps
        old_exp = current_exps[i]
        current_exps[i] = new_exp
        
        score_new, _ = calc_score_from_exponents(current_exps)
        
        # Restore if we reject? 
        # Actually, we just keep current_exps as is if we accept.
        # If we reject, we revert.
        
        if score_new > current_score:
            # Always accept better
            current_score = score_new
            current_n = n_new
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_exps = current_exps[:]
        else:
            # Accept worse with probability exp((score_new - current_score) / T)
            # Note: score is integer, so delta is integer.
            delta_score = score_new - current_score
            if delta_score == 0:
                # Accept to allow exploration
                if random.random() < 0.5:
                    current_score = score_new
                    current_n = n_new
            else:
                # delta_score is negative
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    current_score = score_new
                    current_n = n_new
                else:
                    # Revert
                    current_exps[i] = old_exp
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck? 
            # For now, just keep going.
            
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
