"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    primes = [i for i in range(2, N + 1) if not prime_factors[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    num_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for p in prime_factors[i]:
            mask |= (1 << prime_to_idx[p])
        num_mask[i] = mask
    
    sorted_nums = sorted(range(1, N + 1), key=lambda x: -x)
    n = len(sorted_nums)
    
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_nums[i]
    
    # Greedy initial
    greedy_mask = 0
    greedy_sum = 0
    greedy_set = []
    for num in sorted_nums:
        if num_mask[num] & greedy_mask == 0:
            greedy_set.append(num)
            greedy_mask |= num_mask[num]
            greedy_sum += num
    
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    def search(idx, used_mask, current_sum, selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Loose bound
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            return
        
        # Tighter upper bound
        ub = current_sum
        for j in range(idx, n):
            if num_mask[sorted_nums[j]] & used_mask == 0:
                ub += sorted_nums[j]
        
        if ub <= best_sum:
            return
        
        num = sorted_nums[idx]
        mask = num_mask[num]
        
        # Include first (prefer larger values)
        if mask & used_mask == 0:
            selected.append(num)
            search(idx + 1, used_mask | mask, current_sum + num, selected)
            selected.pop()
        
        # Exclude
        search(idx + 1, used_mask, current_sum, selected)
    
    selected = []
    search(0, 0, 0, selected)
    
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
