import math
import time

T12 = math.factorial(12)
primes = [2, 3, 5, 7, 11]
target_exponents = [10, 5, 2, 1, 1]

def get_divisors():
    divs = [1]
    for p, e in zip(primes, target_exponents):
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * power)
            power *= p
        divs = new_divs
    return divs

all_divs = get_divisors()
# Filter > 12
valid_divs = [d for d in all_divs if d > 12]
valid_divs.sort()

# We also need to quickly find divisors of a number `n` that are in `valid_divs` and <= M.
# Since we are doing DFS, we will iterate through `valid_divs` up to M.
# To speed up, we can precompute a set of valid_divs for O(1) lookup, but we need to iterate in order.
# Actually, in the DFS, we iterate `d` from `min_val` to `M`.
# We can just loop through `valid_divs` and break when `d > M`.

def solve_for_M(M):
    """
    Returns a list of factors if 12! can be decomposed into distinct factors 
    from valid_divs, all <= M, else None.
    """
    # To speed up, we can pre-filter valid_divs to only those <= M
    # But we need to pass this down or filter in the loop.
    
    # Optimization: Use a set for membership? No, we need to iterate.
    # The list `valid_divs` is sorted. We can find the index of the last element <= M.
    # But `min_val` changes.
    
    # Let's just iterate `valid_divs` and check conditions.
    
    def dfs(remaining, min_val, current_factors):
        if remaining == 1:
            return current_factors
        if len(current_factors) >= 12:
            return None
        
        # Pruning: If remaining is prime and remaining < min_val, we can't pick it.
        # Actually, if remaining < min_val, we can't pick any factor >= min_val.
        if remaining < min_val:
            return None
            
        # Iterate through possible next factors d
        # d must be in valid_divs
        # d >= min_val
        # d <= M
        # d divides remaining
        
        for d in valid_divs:
            if d < min_val:
                continue
            if d > M:
                break
            if remaining % d == 0:
                # Try d
                res = dfs(remaining // d, d + 1, current_factors + [d])
                if res is not None:
                    return res
        return None

    return dfs(T12, 13, [])

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Iterate M from smallest to largest
    for M in valid_divs:
        if time.time() > deadline:
            break
        sol = solve_for_M(M)
        if sol is not None:
            # Verify
            prod = 1
            for d in sol:
                prod *= d
            if prod == T12 and len(set(sol)) == len(sol) and all(d > 12 for d in sol):
                return sorted(sol)
            else:
                # Should not happen if logic is correct
                pass
    # Fallback if no solution found within time (should not happen)
    return [T12]

if __name__ == "__main__":
    res = construct()
    print(res)
