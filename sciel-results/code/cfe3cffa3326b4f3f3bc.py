import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence (tau_perp).
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    # We use the first 20 primes. The product of the first 15 is ~6e17.
    # The 16th is 53 (3.4e19 > 1e18). So effectively k <= 15 for square-free.
    # But with exponents, we might use fewer primes with higher powers.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    max_val = 10**18
    
    best_n = 2
    best_score = 0
    
    # Cache for divisor generation to speed up repeated scoring of similar structures?
    # Or just compute directly.
    
    def get_score(primes_list, exponents_list):
        """
        Compute tau_perp for n = product(p_i^e_i).
        """
        if not primes_list:
            return 0
        
        # Generate all divisors
        divisors = [1]
        for p, e in zip(primes_list, exponents_list):
            new_divisors = []
            power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divisors.append(d * power)
                power *= p
            divisors = new_divisors
        
        divisors.sort()
        
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count

    def dfs(idx, max_exp, current_n, current_primes, current_exps):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Evaluate current state if it has at least one prime factor
        if current_primes:
            score = get_score(current_primes, current_exps)
            if score > best_score:
                best_score = score
                best_n = current_n

        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Determine max exponent for this prime
        # current_n * p^e <= max_val
        # p^e <= max_val // current_n
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
            
        # Try exponents from e_max_possible down to 1?
        # Or 1 to e_max_possible?
        # Usually, higher exponents on smaller primes are good for maximizing tau(n),
        # but for tau_perp, we want many distinct primes (square-free-like).
        # However, adding a prime (e=1) increases the divisor set significantly.
        # Let's try all valid exponents.
        
        # Optimization: If we choose e=0, we break the chain of "consecutive" primes
        # in the sense that we can't pick a larger prime later if we enforce non-increasing
        # exponents on the *sequence of primes used*.
        # Wait, the standard DFS for unique factorization with non-increasing exponents
        # assumes we are picking from the list of primes in order.
        # If we skip a prime (e=0), we can still pick later primes?
        # If we enforce e_i >= e_{i+1} for the *chosen* primes, and we skip p_i,
        # then the next chosen prime p_j (j>i) must have e_j <= e_{prev_chosen}.
        # This allows skipping primes.
        # However, the current implementation breaks on e=0.
        # This restricts us to numbers of the form p_1^{e_1} ... p_k^{e_k} where p_i are the first k primes.
        # Is it ever beneficial to skip a small prime?
        # e.g. 2^a * 5^b vs 2^a * 3^c * 5^b?
        # Adding 3 usually increases the number of coprime pairs because divisors with 3 and without 3 are coprime in that component.
        # So skipping a small prime is likely suboptimal.
        # We will stick to the "consecutive primes from 2" assumption for now, but allow e=0 to terminate.
        
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            dfs(idx + 1, e, next_n, current_primes + [p], current_exps + [e])
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    dfs(0, 60, 1, [], [])
    
    # Also try some specific hand-crafted candidates or local improvements?
    # The DFS above explores the space of numbers with non-increasing exponents on consecutive primes.
    # This covers the most likely optimal structures.
    
    # One specific case: The primorial P_15 = 2*3*...*47
    # This has e_i = 1 for all i.
    # The DFS will find this.
    
    return best_n

if __name__ == "__main__":
    n = construct()
    # print(n)
