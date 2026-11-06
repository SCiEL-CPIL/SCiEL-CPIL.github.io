import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    n_valid = len(valid_divs)
    
    # Precompute suffix products for valid_divs
    # suffix_prod[i] = product(valid_divs[i:])
    suffix_prod = [1] * (n_valid + 1)
    for i in range(n_valid - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i+1] * valid_divs[i]
        
    # We want to find the minimal M such that there exists a subset of 
    # valid_divs <= M that multiplies to target.
    # Iterate M in increasing order.
    
    # To speed up, we can iterate over the indices of valid_divs as the potential M.
    # For a given index m_idx, M = valid_divs[m_idx].
    # Candidates are valid_divs[0 ... m_idx].
    
    # DFS parameters:
    # start_idx: index in valid_divs to start picking from
    # remaining: product to achieve
    # count: number of factors picked (limit 12)
    # current_factors: list of picked factors
    
    # We use a non-recursive or iterative approach? Recursion depth is max 12, so recursion is safe.
    
    def dfs(start_idx, m_idx, remaining, count, current_factors):
        if time.time() - start_time > timeout:
            return None
            
        if remaining == 1:
            if count > 0:
                return list(current_factors)
            return None
            
        if count >= 12:
            return None
            
        # Pruning: 
        # 1. If the product of all available candidates from start_idx to m_idx is less than remaining, fail.
        # The available candidates are valid_divs[start_idx : m_idx+1]
        # We need the product of this slice.
        # We can precompute prefix products? No, suffix products are easier.
        # Product(valid_divs[start_idx : m_idx+1]) = suffix_prod[start_idx] / suffix_prod[m_idx+1]
        # But integer division might be lossy if not divisible? 
        # Since valid_divs are divisors of target, and we are just multiplying them, 
        # suffix_prod[start_idx] is divisible by suffix_prod[m_idx+1] because 
        # suffix_prod[start_idx] = valid_divs[start_idx] * ... * valid_divs[m_idx] * suffix_prod[m_idx+1]
        # So division is exact.
        
        avail_prod = suffix_prod[start_idx] // suffix_prod[m_idx+1]
        if avail_prod < remaining:
            return None
            
        # 2. If remaining is prime, it must be a single factor.
        # Check if remaining is in valid_divs[start_idx : m_idx+1]
        # Actually, if remaining is prime, it must be 2,3,5,7,11. But factors must be > 12.
        # So if remaining is prime, and remaining <= 12, it's impossible.
        # But remaining is a divisor of 12!, so if it's prime, it's one of 2,3,5,7,11.
        # None of these are > 12. So if remaining is prime, we can never pick it as a factor > 12.
        # Wait, if remaining is prime, we need a factor f such that f divides remaining.
        # The only divisors of a prime p are 1 and p.
        # f must be > 12. So f must be p.
        # But p is 2,3,5,7,11 which are <= 12.
        # So if remaining is prime, we CANNOT form it with factors > 12.
        # Thus, return None.
        
        # How to check if remaining is prime?
        # Since remaining is a divisor of 12!, its prime factors are in {2,3,5,7,11}.
        # We can just check if it has a divisor > 1 and < remaining.
        # Or simpler: if remaining is in the set of primes {2,3,5,7,11}, return None.
        # But what if remaining is composite but cannot be formed? 
        # The DFS will handle that.
        # The prime check is a strong early cut.
        
        # Let's add a quick check for small primes
        if remaining in (2, 3, 5, 7, 11):
            return None
            
        # Iterate candidates
        # We can restrict the upper bound of i.
        # candidates[i] must be <= remaining.
        # Also, we must pick at least 1 factor.
        
        # Optimization: If remaining is a divisor of the product of the largest few candidates?
        
        for i in range(start_idx, m_idx + 1):
            c = valid_divs[i]
            
            if c > remaining:
                # Since valid_divs is sorted, no further candidate can divide remaining
                # (unless remaining is 0, which is not possible)
                # Actually, if c > remaining, c cannot divide remaining (integer division would be 0, 
                # but we check c <= remaining).
                break
                
            if remaining % c == 0:
                current_factors.append(c)
                res = dfs(i + 1, m_idx, remaining // c, count + 1, current_factors)
                current_factors.pop()
                if res:
                    return res
                    
        return None

    # Iterate M
    for m_idx in range(n_valid):
        M = valid_divs[m_idx]
        
        # Quick check: Is M large enough?
        # The smallest possible product of 1 factor is M.
        # The target is 12!.
        # If M < 12! (which is always true for M < target), we need multiple factors.
        
        # Start DFS
        sol = dfs(0, m_idx, target, 0, [])
        
        if sol:
            return sol
            
        if time.time() - start_time > timeout:
            break
            
    # Fallback
    return [target]
