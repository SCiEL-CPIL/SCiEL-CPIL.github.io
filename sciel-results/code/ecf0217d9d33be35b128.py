import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    # LCM of 1..30
    L = reduce(lcm, range(1, 31))
    
    # Precompute scaled weights
    # weight[i] corresponds to coefficient of 1/(i+1)
    # Sum = sum(c[i] * weight[i]) / L
    weights = [L // (i + 1) for i in range(30)]
    
    n = 30
    half = 15
    
    # Split weights into two halves
    w1 = weights[:half]
    w2 = weights[half:]
    
    def generate_sums(weights_subset):
        """
        Generate all possible sums for the given weights.
        Returns a list of sums.
        """
        sums = [0]
        for w in weights_subset:
            new_sums = []
            # Optimization: local variables
            append = new_sums.append
            for s in sums:
                append(s - w)
                append(s)
                append(s + w)
            sums = new_sums
        return sums
    
    # Generate sums for first half
    sums1 = generate_sums(w1)
    
    # Generate sums for second half
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2.sort()
    
    best_abs_k = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    # Find the minimal nonzero absolute total sum
    # Total sum K = s1 + s2
    # We want min |s1 + s2| such that s1 + s2 != 0
    
    for s1 in sums1:
        # We want s2 such that s2 is close to -s1
        target = -s1
        
        # Find insertion point in sums2
        idx = bisect_left(sums2, target)
        
        # Check neighbors: idx-1, idx, idx+1
        # It's possible that the closest value is further away if there are gaps,
        # but since we are minimizing absolute difference, the closest value in a sorted list
        # to a target must be one of the immediate neighbors of the insertion point.
        
        candidates = []
        if idx > 0:
            candidates.append(sums2[idx - 1])
        if idx < len(sums2):
            candidates.append(sums2[idx])
        if idx + 1 < len(sums2):
            candidates.append(sums2[idx + 1])
            
        # Also check idx-2 and idx+2 just in case of duplicates or specific distributions?
        # No, for a sorted array, the minimum absolute difference to a target is always
        # achieved by the predecessor or successor of the insertion point.
        
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_k:
                best_abs_k = abs_total
                best_s1 = s1
                best_s2 = s2
                
    # If no nonzero sum found (impossible unless weights are 0, which they aren't), 
    # but theoretically if all sums were 0? No, weights are distinct non-zero.
    # However, it is possible that the *only* sums available are 0?
    # With 30 variables, there are 3^30 sums. The range is large. 0 is just one point.
    # There will be many non-zero sums.
    
    # Now we need to reconstruct the coefficients.
    # We need c1 such that sum(c1[i] * w1[i]) == best_s1
    # We need c2 such that sum(c2[i] * w2[i]) == best_s2
    
    def find_coeffs(weights_subset, target_sum):
        """
        Find a vector of coefficients in {-1, 0, 1} that sums to target_sum using the given weights.
        Uses DFS with pruning.
        """
        n_sub = len(weights_subset)
        coeffs = [0] * n_sub
        
        # Precompute suffix sums of absolute weights for pruning
        # max_possible[i] = sum of weights[i:]
        suffix_sums = [0] * (n_sub + 1)
        for i in range(n_sub - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights_subset[i]
        
        def dfs(pos, current_sum):
            if pos == n_sub:
                return current_sum == target_sum
            
            w = weights_subset[pos]
            
            # Pruning:
            # The maximum change we can make with remaining elements is suffix_sums[pos+1]
            # So if |target_sum - (current_sum + w)| > suffix_sums[pos+1], we can prune +w case
            # Similarly for 0 and -w
            
            # Try 1
            new_sum = current_sum + w
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = 1
                if dfs(pos + 1, new_sum):
                    return True
            
            # Try 0
            new_sum = current_sum
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = 0
                if dfs(pos + 1, new_sum):
                    return True
            
            # Try -1
            new_sum = current_sum - w
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = -1
                if dfs(pos + 1, new_sum):
                    return True
            
            coeffs[pos] = 0
            return False
        
        if dfs(0, 0):
            return coeffs
        return None

    coeffs1 = find_coeffs(w1, best_s1)
    coeffs2 = find_coeffs(w2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        # Fallback: This should not happen if best_s1 and best_s2 are valid sums from our generation.
        # But if it does, return a valid non-zero solution.
        # Simple fallback: c[0]=1, rest 0 -> sum = 1/1 = 1.
        return [1] + [0] * 29
        
    return coeffs1 + coeffs2
