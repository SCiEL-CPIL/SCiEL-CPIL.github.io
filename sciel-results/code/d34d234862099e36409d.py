# EVOLVE-BLOCK-START
import itertools
from math import isqrt

def _generate_smooth_numbers(primes, limit):
    """Generate all S-smooth numbers up to limit."""
    smooth = set([1])
    for p in primes:
        new_smooth = set()
        for s in smooth:
            val = s
            while val <= limit:
                new_smooth.add(val)
                val *= p
        smooth |= new_smooth
    return sorted([s for s in smooth if 1 <= s <= limit])

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes_in_sums(values):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            primes |= _factorize(s)
    return len(primes)

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the number of distinct
    primes dividing all pairwise sums (of distinct elements) is minimized.
    
    Strategy: Model as constraint satisfaction. For a candidate prime set S,
    generate S-smooth numbers up to 10000 (for elements) and up to 20000 (for sums).
    Use backtracking to find 12 elements whose pairwise sums are all S-smooth.
    Try prime sets in order of increasing size to find the smallest S.
    """
    
    # Candidate prime sets to try, in order of increasing size
    candidate_prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    ]
    
    for primes in candidate_prime_sets:
        # Generate S-smooth numbers up to 10000 (for elements) and 20000 (for sums)
        smooth_elements = _generate_smooth_numbers(primes, 10000)
        smooth_sums_set = set(_generate_smooth_numbers(primes, 20000))
        
        # We need to find 12 distinct elements from smooth_elements
        # such that all pairwise sums are in smooth_sums_set
        
        n_needed = 12
        solution = []
        
        # Backtracking search with pruning
        def backtrack(start_idx, current_indices):
            if len(current_indices) == n_needed:
                return True
            
            remaining = n_needed - len(current_indices)
            # Pruning: not enough candidates left
            if len(smooth_elements) - start_idx < remaining:
                return False
            
            for i in range(start_idx, len(smooth_elements)):
                candidate = smooth_elements[i]
                # Check if candidate is compatible with all already chosen
                valid = True
                for j in current_indices:
                    if (candidate + smooth_elements[j]) not in smooth_sums_set:
                        valid = False
                        break
                
                if valid:
                    current_indices.append(i)
                    if backtrack(i + 1, current_indices):
                        return True
                    current_indices.pop()
            
            return False
        
        if backtrack(0, []):
            # Reconstruct solution
            # We need to track the actual indices chosen
            # Let's redo with proper tracking
            pass
        
        # Redo with proper index tracking
        chosen_indices = []
        
        def backtrack2(start_idx):
            if len(chosen_indices) == n_needed:
                return True
            
            remaining = n_needed - len(chosen_indices)
            if len(smooth_elements) - start_idx < remaining:
                return False
            
            for i in range(start_idx, len(smooth_elements)):
                candidate = smooth_elements[i]
                valid = True
                for j in chosen_indices:
                    if (candidate + smooth_elements[j]) not in smooth_sums_set:
                        valid = False
                        break
                
                if valid:
                    chosen_indices.append(i)
                    if backtrack2(i + 1):
                        return True
                    chosen_indices.pop()
            
            return False
        
        if backtrack2(0):
            return [smooth_elements[i] for i in chosen_indices]
    
    # Fallback: use a simple construction
    # Use multiples of a highly composite number to minimize prime variety
    # 12 distinct multiples of 2520 (LCM of 1..10) up to 10000
    base = 2520
    result = []
    for k in range(1, 13):
        val = base * k
        if val <= 10000:
            result.append(val)
        else:
            break
    # If we don't have 12, use smaller multiples
    if len(result) < 12:
        result = []
        for k in range(1, 13):
            val = 504 * k  # LCM of 1..9 / 2
            if val <= 10000:
                result.append(val)
    
    # Ensure we have exactly 12 distinct values
    if len(result) < 12:
        # Pad with small multiples
        used = set(result)
        k = 1
        while len(result) < 12:
            val = 100 * k
            if val not in used and val <= 10000:
                result.append(val)
                used.add(val)
            k += 1
    
    return result[:12]
# EVOLVE-BLOCK-END
