"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(0) for _ in range(n + 1)]
    for i in range(1, n + 1):
        reciprocals[i] = Fraction(1, i)
    
    # Greedy solution to get a good initial bound
    def greedy_solution():
        selected = set()
        for x in range(1, n + 1):
            can_add = True
            for a in selected:
                if a < x:
                    s = a + x
                    if s % 2 == 0:
                        b = s // 2
                        if a < b < x and b in selected:
                            can_add = False
                            break
            if can_add:
                selected.add(x)
        return selected
    
    initial_selected = greedy_solution()
    initial_sum = sum(reciprocals[x] for x in initial_selected)
    
    # Convert initial solution to bitset
    initial_bits = 0
    for x in initial_selected:
        initial_bits |= (1 << x)
    
    # Branch and bound with bitsets
    # We process numbers in increasing order: 1, 2, ..., 256
    # State: x (next number to consider), selected_bits, forbidden_bits, current_sum (float for speed)
    
    # Precompute suffix sums of reciprocals for upper bound
    suffix = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix[i] = suffix[i + 1] + 1.0 / i
    
    best_sum = float(initial_sum)
    best_bits = initial_bits
    
    # For efficiency, we'll use float sums during search and only compute exact sum at the end
    # But we need to be careful with precision. Let's use a threshold.
    
    def dfs(x, selected_bits, forbidden_bits, current_sum):
        nonlocal best_sum, best_bits
        
        # Upper bound: current sum + sum of all remaining non-forbidden elements
        # We can compute this more tightly by checking which elements are forbidden
        # But for speed, let's use the suffix sum as a loose upper bound first
        if current_sum + suffix[x] <= best_sum:
            return
        
        if x > n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_bits = selected_bits
            return
        
        # Option 1: Take x (if not forbidden)
        if not (forbidden_bits >> x) & 1:
            # Check if adding x creates a 3-AP with two already selected elements
            # For any a in selected with a < x, check if (a+x)//2 is in selected
            can_add = True
            sb = selected_bits
            while sb:
                lsb = sb & -sb
                a = lsb.bit_length() - 1
                if a < x:
                    s = a + x
                    if s % 2 == 0:
                        b = s // 2
                        if a < b < x and (selected_bits >> b) & 1:
                            can_add = False
                            break
                sb -= lsb
            
            if can_add:
                # Compute new forbidden bits
                new_forbidden = forbidden_bits
                sb = selected_bits
                while sb:
                    lsb = sb & -sb
                    a = lsb.bit_length() - 1
                    if a < x:
                        c = 2 * x - a
                        if x < c <= n:
                            new_forbidden |= (1 << c)
                    sb -= lsb
                
                # Also, x itself might forbid some elements based on pairs in selected
                # But we already handled that above
                
                new_sum = current_sum + 1.0 / x
                if new_sum + suffix[x + 1] > best_sum:
                    dfs(x + 1, selected_bits | (1 << x), new_forbidden, new_sum)
        
        # Option 2: Skip x
        dfs(x + 1, selected_bits, forbidden_bits, current_sum)
    
    dfs(1, 0, 0, 0.0)
    
    # Extract the best solution
    result = [i for i in range(1, n + 1) if (best_bits >> i) & 1]
    return result
# EVOLVE-BLOCK-END
