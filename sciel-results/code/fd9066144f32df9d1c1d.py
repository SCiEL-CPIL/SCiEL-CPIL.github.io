import math
import time

def construct():
    T12 = math.factorial(12)
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    all_divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        mult = 1
        for _ in range(e + 1):
            for d in all_divisors:
                new_divs.append(d * mult)
            mult *= p
        all_divisors = new_divs
        
    # Filter divisors > 12
    valid_divisors = sorted([d for d in all_divisors if d > 12])
    
    # Map divisor to exponent vector
    def get_exp_vec(n):
        vec = [0] * 5
        for i, p in enumerate(primes):
            while n % p == 0:
                n //= p
                vec[i] += 1
        return tuple(vec)
        
    divisor_exp = {d: get_exp_vec(d) for d in valid_divisors}
    
    # Precompute value from exp vector for quick check? Or just use integer division.
    # Using integer division is fast enough.
    
    def solve_for_M(M):
        """
        Returns a list of factors if a valid decomposition exists with all factors <= M, else None.
        """
        # State: rem_exp (tuple), start_idx (int), current_factors (list)
        # We use DFS with backtracking
        
        def dfs(rem_exp, start_idx, current_factors):
            if all(e == 0 for e in rem_exp):
                return current_factors[:]
                
            if len(current_factors) >= 12:
                return None
                
            # Calculate remaining value
            rem_val = 1
            for i, e in enumerate(rem_exp):
                rem_val *= primes[i] ** e
                
            if rem_val == 1:
                return current_factors[:]
                
            # Iterate possible next factors
            for i in range(start_idx, len(valid_divisors)):
                d = valid_divisors[i]
                
                # Prune: d must be <= M
                if d > M:
                    break
                    
                ev = divisor_exp[d]
                
                # Check if d fits in rem_exp
                fits = True
                for j in range(5):
                    if rem_exp[j] < ev[j]:
                        fits = False
                        break
                if not fits:
                    continue
                    
                # Check if d divides rem_val (implied by exp check, but rem_val % d == 0 is a good sanity check)
                if rem_val % d != 0:
                    continue
                
                new_rem_val = rem_val // d
                new_rem_exp = tuple(rem_exp[j] - ev[j] for j in range(5))
                
                # Pruning:
                if new_rem_val == 1:
                    # Success if we can form the rest (which is nothing)
                    return current_factors + [d]
                else:
                    # We need to form new_rem_val with factors from indices > i, all <= M
                    # The smallest available factor is valid_divisors[i+1]
                    if i + 1 >= len(valid_divisors):
                        continue # No more factors available, but new_rem_val > 1
                    
                    next_d = valid_divisors[i+1]
                    if next_d > M:
                        continue # No more factors <= M available
                    
                    # If the smallest available factor is greater than new_rem_val, 
                    # then no factor can divide new_rem_val (since all subsequent factors are larger)
                    if next_d > new_rem_val:
                        continue
                        
                    # Recurse
                    res = dfs(new_rem_exp, i + 1, current_factors + [d])
                    if res is not None:
                        return res
            return None
            
        return dfs(tuple(exponents), 0, [])

    # Search for minimum M
    # Lower bound: 13
    # Upper bound: T12 (single factor)
    # We can binary search or linear search.
    # Linear search from 13 upwards is likely fast because the optimal M is small.
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11.
    # Let's try linear.
    
    best_solution = None
    for M in range(13, T12 + 1):
        # Optimization: If M is small, the number of valid divisors is small.
        # We can break early if we find a solution.
        sol = solve_for_M(M)
        if sol is not None:
            best_solution = sol
            break
            
    if best_solution is None:
        # Fallback
        return [T12]
        
    return best_solution

# Test locally?
# print(construct())
