import math
from itertools import product

def construct():
    N = math.factorial(12)
    primes = [2, 3, 5, 7, 11]
    
    # Get prime factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    divisors = []
    # Iterate over all combinations of exponents
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append((d, [e2, e3, e5, e7, e11]))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    # We iterate through possible maximum values M
    # For each M, we check if there is a valid subset from divisors <= M
    # To optimize, we can just iterate through the sorted divisors list
    # and treat divisors[i] as the current M
    
    n_divs = len(divisors)
    
    # Backtracking function to find a subset of divisors[0..idx] 
    # that sums to target_exp, using at most 12 elements.
    # We want to find ANY such subset.
    
    def find_subset(idx, current_exp, count, path):
        # Pruning: if count > 12, return False
        if count > 12:
            return False
        
        # Check if current_exp equals target_exp
        if current_exp == target_exp:
            # We found a valid subset
            # We need to ensure all elements in path are distinct (they are, as we pick from list)
            # and > 12 (ensured by generation)
            return True
        
        # If idx < 0, we can't add more elements
        if idx < 0:
            return False
        
        # Option 1: Do not pick divisors[idx]
        if find_subset(idx - 1, current_exp, count, path):
            return True
            
        # Option 2: Pick divisors[idx]
        # Check if adding divisors[idx] exceeds target_exp
        d_val, d_exp = divisors[idx]
        new_exp = [current_exp[i] + d_exp[i] for i in range(5)]
        if all(new_exp[i] <= target_exp[i] for i in range(5)):
            path.append(d_val)
            if find_subset(idx - 1, new_exp, count + 1, path):
                return True
            path.pop()
            
        return False

    # Iterate M from smallest to largest
    # We check if a solution exists using only divisors up to index i
    for i in range(n_divs):
        M = divisors[i][0]
        # We search in divisors[0..i]
        path = []
        # Optimization: If the product of the smallest 12 divisors in the range is > N, 
        # or product of largest 1 is < N, we might prune? 
        # But the backtracking handles it.
        
        # To speed up, we can sort the candidates for this M in descending order?
        # Or just use the sorted list as is.
        # Using ascending order (small to large) might find solutions with smaller max faster?
        # Actually, we are fixing M, so all are <= M.
        
        # Let's try to find a solution.
        # To make backtracking faster, we can try larger factors first?
        # If we pick larger factors, we reach the target product faster, reducing depth.
        # So let's reverse the search order: try divisors[i], divisors[i-1], ...
        
        # Modify find_subset to search from i down to 0.
        # The function above does this.
        
        if find_subset(i, [0, 0, 0, 0, 0], 0, path):
            return path
            
    # If no solution found (should not happen, [N] is a solution)
    return [N]
