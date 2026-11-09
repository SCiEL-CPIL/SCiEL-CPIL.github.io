"""Combined approach: prime-power baseline with greedy merge refinement."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_START_TIME = time.time()
_MAX_TIME_LIMIT = 85.0

def construct():
    """
    Returns a list of distinct integers from 1 to 256, pairwise coprime,
    maximizing their sum.
    """
    # Check time limit to ensure we don't get killed by the evaluator
    if time.time() - _START_TIME > _MAX_TIME_LIMIT:
        return [1]
    
    limit = 256
    
    # Identify all prime numbers up to 256
    primes_list = []
    is_prime_flag = [True] * (limit + 1)
    is_prime_flag[0] = False
    is_prime_flag[1] = False
    
    for current in range(2, int(limit ** 0.5) + 1):
        if is_prime_flag[current]:
            for multiple in range(current * current, limit + 1, current):
                is_prime_flag[multiple] = False
    
    for num in range(2, limit + 1):
        if is_prime_flag[num]:
            primes_list.append(num)
            
    # Construct the optimal set:
    # 1 is always included.
    # For each prime p, include the largest power p^k <= 256.
    # These numbers are pairwise coprime because each is a power of a unique prime.
    current_set = [1]
    
    for p in primes_list:
        max_power_val = p
        # Find largest p^k <= 256
        while max_power_val * p <= limit:
            max_power_val *= p
        current_set.append(max_power_val)
    
    current_sum = sum(current_set)
    best_set = list(current_set)
    best_sum = current_sum
    
    # Greedy merge loop:
    # We want to maximize sum by merging coprime pairs a,b into a*b
    # if a*b <= 256 and a*b not in set.
    # Gain is a*b - a - b.
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_pair = None
        
        # Sort to allow early termination in inner loop
        s = sorted(current_set)
        for i in range(len(s)):
            if s[i] > 256:
                continue
            for j in range(i + 1, len(s)):
                prod = s[i] * s[j]
                if prod > 256:
                    break
                if prod in current_set:
                    continue
                gain = prod - s[i] - s[j]
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (s[i], s[j])
        
        if best_pair:
            a, b = best_pair
            # Remove a and b
            current_set.remove(a)
            current_set.remove(b)
            current_set.append(a * b)
            current_sum += best_gain
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            
            improved = True
            
        # Time check
        if time.time() - _START_TIME > _MAX_TIME_LIMIT - 5:
            break
            
    # Verify pairwise coprime (sanity check)
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(best_set[i], best_set[j]) != 1:
                # Fallback to prime powers if something went wrong
                fallback = [1]
                for p in primes_list:
                    val = p
                    while val * p <= limit:
                        val *= p
                    fallback.append(val)
                return [int(x) for x in fallback]
    
    # Ensure all elements are standard Python integers
    final_result = []
    for val in best_set:
        final_result.append(int(val))
        
    return final_result
# EVOLVE-BLOCK-END
