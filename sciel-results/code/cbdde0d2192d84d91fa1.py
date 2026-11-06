"""Improved Egyptian fraction search with A*-style priority queue."""
import time
from math import gcd
import heapq

def construct():
    deadline = time.time() + 80  # 80s wall-clock guard
    
    # First, try to find a solution with k=8 (theoretical minimum for denominators >= 5)
    # If that's too slow, try k=9, k=10, etc.
    
    def heuristic(num, den, k, min_d):
        """
        Admissible heuristic: minimum number of terms needed.
        Each term is at most 1/min_d, so we need at least ceil(num*den/k... 
        Actually, each term 1/d <= 1/min_d, so k terms sum to at most k/min_d.
        We need num/den <= k/min_d, so k >= num*min_d/den.
        Also, the sum of the k largest possible terms (starting from min_d) 
        is an upper bound on what k terms can achieve.
        """
        if num == 0:
            return 0
        # Lower bound: k >= ceil(num * min_d / den)
        lb = (num * min_d + den - 1) // den
        # Also, k >= 1 if num > 0
        if num > 0:
            lb = max(lb, 1)
        return lb
    
    def search(k_target, min_d_start=5):
        """
        BFS/A* search for exactly k_target denominators.
        State: (num, den, min_d, path)
        We want to find a path of length k_target that sums to 1.
        """
        # Use a priority queue: (heuristic_estimate, num, den, min_d, path)
        # Start: need sum = 1/1 with k_target terms, min denominator = 5
        start = (1, 1, min_d_start, [])
        
        # We'll do a breadth-first search by depth (number of terms chosen)
        # At each level, we choose one more denominator.
        
        # For efficiency, we track visited states (num, den, min_d) to avoid reprocessing
        visited = set()
        
        # Level 0: no denominators chosen, remaining sum = 1/1
        current_level = [(1, 1, min_d_start, [])]
        
        for depth in range(k_target):
            next_level = []
            for (num, den, min_d, path) in current_level:
                if time.time() > deadline:
                    return None
                
                # If this is the last term, we need 1/d = num/den
                if depth == k_target - 1:
                    # num/den = 1/d => d = den/num
                    # d must be integer, so num must divide den
                    if num == 0:
                        continue
                    if den % num == 0:
                        d = den // num
                        if d >= min_d and d <= 10000000 and d not in path:
                            return path + [d]
                    continue
                
                # Determine range for next denominator d
                # 1/d <= num/den => d >= den/num
                # Also d >= min_d
                if num <= 0:
                    continue
                
                d_min = max(min_d, (den + num - 1) // num)
                
                # Upper bound: after picking d, remaining sum is num/den - 1/d
                # The remaining k_target - depth - 1 terms must each be at least 1/(d+1), 1/(d+2), ...
                # Actually, they must be >= 1/d' for d' > d, so each is < 1/d
                # Sum of remaining (k_target - depth - 1) terms < (k_target - depth - 1)/d
                # So num/den - 1/d < (k_target - depth - 1)/d
                # num/den < (k_target - depth)/d
                # d < (k_target - depth) * den / num
                d_max = ((k_target - depth) * den) // num
                d_max = min(d_max, 10000000)
                
                if d_max < d_min:
                    continue
                
                # Also, the remaining sum after picking d must be positive
                # num/den - 1/d > 0 => num*d > den
                if num * d_min <= den:
                    d_min = den // num + 1
                    if d_min > d_max:
                        continue
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        return None
                    
                    # New remaining: num/den - 1/d = (num*d - den)/(den*d)
                    new_num = num * d - den
                    new_den = den * d
                    
                    if new_num <= 0:
                        continue
                    
                    # Reduce
                    g = gcd(new_num, new_den)
                    new_num //= g
                    new_den //= g
                    
                    state_key = (new_num, new_den, d + 1)
                    if state_key in visited:
                        continue
                    visited.add(state_key)
                    
                    next_level.append((new_num, new_den, d + 1, path + [d]))
            
            current_level = next_level
            if not current_level:
                return None
        
        return None
    
    # Try increasing k
    for k in range(8, 13):
        if time.time() > deadline:
            break
        result = search(k)
        if result is not None:
            # Verify the solution
            total = 0
            for d in result:
                total = total * d + 1
                total_den = 1
            # Actually, let's verify properly
            from fractions import Fraction
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == Fraction(1, 1) and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: greedy algorithm
    def greedy_egyptian():
        result = []
        num, den = 1, 1
        min_d = 5
        while num > 0 and len(result) <= 256:
            d_min = max(min_d, (den + num - 1) // num)
            if d_min > 10000000:
                return None
            result.append(d_min)
            num = num * d_min - den
            den = den * d_min
            g = gcd(num, den)
            num //= g
            den //= g
            min_d = d_min + 1
        if num == 0 and 1 <= len(result) <= 256:
            from fractions import Fraction
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == Fraction(1, 1):
                return result
        return None
    
    result = greedy_egyptian()
    if result is not None:
        return result
    
    raise RuntimeError("no representation found")
