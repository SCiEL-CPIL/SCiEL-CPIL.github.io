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
    
    num_divs = len(divisors)  # 60 divisors
    
    # For small k (up to 6), use meet-in-the-middle
    # For larger k, use greedy + refinement
    
    # Try k = 1, 2, 3, ... in order
    # k=1: check if target is a divisor
    if target in divisors:
        return [target]
    
    # k=2: check all pairs
    div_set = set(divisors)
    for i in range(num_divs):
        complement = target - divisors[i]
        if complement > divisors[i] and complement in div_set:
            return [divisors[i], complement]
    
    # k=3: check all triples (60 choose 3 = 34220, fast)
    for i in range(num_divs):
        for j in range(i+1, num_divs):
            s2 = divisors[i] + divisors[j]
            if s2 >= target:
                break
            complement = target - s2
            if complement > divisors[j] and complement in div_set:
                # Check complement is not already used
                if complement != divisors[i] and complement != divisors[j]:
                    return [divisors[i], divisors[j], complement]
    
    # k=4: 60 choose 4 = 487635, still manageable
    for i in range(num_divs):
        for j in range(i+1, num_divs):
            s2 = divisors[i] + divisors[j]
            if s2 >= target:
                break
            for k in range(j+1, num_divs):
                s3 = s2 + divisors[k]
                if s3 >= target:
                    break
                complement = target - s3
                if complement > divisors[k] and complement in div_set:
                    if complement not in (divisors[i], divisors[j], divisors[k]):
                        return [divisors[i], divisors[j], divisors[k], complement]
    
    # k=5: 60 choose 5 = 5.4M, might be slow. Use meet-in-the-middle.
    # Split divisors into two halves
    half = num_divs // 2
    first_half = divisors[:half]
    second_half = divisors[half:]
    
    # For k=5: 2 from first, 3 from second, or 3 from first, 2 from second
    # Generate all sums of 2 from first half, 3 from second half, etc.
    
    def generate_comb_sums(arr, r):
        """Generate all sums of r elements from arr with the indices."""
        results = {}
        def backtrack(start, remaining, current_sum, current_indices):
            if remaining == 0:
                if current_sum not in results or len(current_indices) < len(results[current_sum]):
                    results[current_sum] = list(current_indices)
                return
            for i in range(start, len(arr)):
                if current_sum + arr[i] > target:  # prune
                    break
                current_indices.append(i)
                backtrack(i + 1, remaining - 1, current_sum + arr[i], current_indices)
                current_indices.pop()
        backtrack(0, r, 0, [])
        return results
    
    # k=5: try (2,3) and (3,2) splits
    # But we need to ensure no overlap. Let's just do full search for k=5 with pruning.
    # 60 choose 5 = 5,461,512 - feasible in Python with pruning
    
    found = False
    for i in range(num_divs):
        if time.time() > deadline - 10:
            break
        for j in range(i+1, num_divs):
            s2 = divisors[i] + divisors[j]
            if s2 >= target:
                break
            for k in range(j+1, num_divs):
                s3 = s2 + divisors[k]
                if s3 >= target:
                    break
                for l in range(k+1, num_divs):
                    s4 = s3 + divisors[l]
                    if s4 >= target:
                        break
                    complement = target - s4
                    if complement > divisors[l] and complement in div_set:
                        if complement not in (divisors[i], divisors[j], divisors[k], divisors[l]):
                            return [divisors[i], divisors[j], divisors[k], divisors[l], complement]
    
    # k=6: 60 choose 6 = 50M, too slow for full search. Use greedy + local search.
    # Greedy: take largest possible divisors
    def greedy_solve():
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
        return None
    
    greedy_result = greedy_solve()
    if greedy_result:
        return greedy_result
    
    # This shouldn't happen for practical numbers, but as fallback
    # Use a more exhaustive search
    # For safety, return a valid solution even if not minimal
    # Since 5040 is practical, a solution must exist
    
    # Try a different greedy: use all divisors except those summing to (5040 - target)
    diff = n - target  # 61
    # Find divisors summing to 61
    # 61 is prime, divisors of 5040 that could sum to 61:
    # Try small combinations
    small_divs = [d for d in divisors if d <= 61]
    for r in range(1, len(small_divs)+1):
        for comb in combinations(small_divs, r):
            if sum(comb) == 61:
                # Use all divisors except those in comb
                result = [d for d in divisors if d not in set(comb)]
                return result
        if time.time() > deadline:
            break
    
    # Ultimate fallback
    return greedy_result if greedy_result else divisors
# EVOLVE-BLOCK-END
