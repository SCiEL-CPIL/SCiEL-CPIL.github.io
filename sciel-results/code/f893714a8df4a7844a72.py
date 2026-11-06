"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    num_divs = len(divisors)
    
    # Split divisors into two halves for Meet-in-the-Middle
    half = num_divs // 2
    first_half = divisors[:half]
    second_half = divisors[half:]
    
    def get_comb_sums(arr, r, target_limit):
        """
        Returns a dictionary mapping sum -> tuple of values.
        Only keeps one tuple per sum.
        """
        sums_dict = {}
        if r == 0:
            return {0: ()}
        
        # Use itertools.combinations for efficiency
        for comb in combinations(arr, r):
            s = sum(comb)
            if s <= target_limit:
                if s not in sums_dict:
                    sums_dict[s] = comb
        return sums_dict

    # We want to minimize k.
    # Try k from 1 to 10
    for k in range(1, 11):
        if time.time() > deadline - 5.0:
            break
            
        found = False
        # i is number of elements from first_half
        # j is number of elements from second_half
        # i + j = k
        min_i = max(0, k - len(second_half))
        max_i = min(k, len(first_half))
        
        for i in range(min_i, max_i + 1):
            j = k - i
            
            # Generate sums from first half
            sums1 = get_comb_sums(first_half, i, target)
            if not sums1:
                continue
                
            # Generate sums from second half
            sums2 = get_comb_sums(second_half, j, target)
            if not sums2:
                continue
            
            # Check for match
            # Iterate over smaller dict
            if len(sums1) < len(sums2):
                for s1, comb1 in sums1.items():
                    s2_needed = target - s1
                    if s2_needed in sums2:
                        comb2 = sums2[s2_needed]
                        result = list(comb1) + list(comb2)
                        # Verify sum and distinctness
                        if sum(result) == target and len(set(result)) == len(result):
                            return result
            else:
                for s2, comb2 in sums2.items():
                    s1_needed = target - s2
                    if s1_needed in sums1:
                        comb1 = sums1[s1_needed]
                        result = list(comb1) + list(comb2)
                        if sum(result) == target and len(set(result)) == len(result):
                            return result
                            
    # Fallback if no solution found for k <= 10
    # Greedy
    rem = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= rem:
            chosen.append(d)
            rem -= d
        if rem == 0:
            break
    if rem == 0:
        return chosen
        
    return []
# EVOLVE-BLOCK-END
