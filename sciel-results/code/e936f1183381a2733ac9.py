import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    N = 512
    K = 32
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set (bit x-1 for x)."""
        count = 0
        # d ranges 1..255 (since max element 512, 2d <= 511)
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count

    def set_to_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def bitset_to_set(b):
        s = []
        for i in range(1, N + 1):
            if (b >> (i - 1)) & 1:
                s.append(i)
        return s

    # Pre-compute for each number x in 1..512, the set of differences d 
    # for which x can be part of a 3-AP (a, a+d, a+2d) within [1, 512].
    # Specifically, for a given x, what d values can x support as a, a+d, or a+2d?
    # x=a: d such that x+2d <= 512 => d <= (512-x)//2
    # x=a+d: d such that x-d >= 1 and x+d <= 512 => d <= min(x-1, 512-x)
    # x=a+2d: d such that x-2d >= 1 => d <= (x-1)//2
    # Actually, we just need to know if adding x helps complete any AP.
    # A faster heuristic: 
    # If we have a current set S, adding x creates a new d if there exists 
    # y, z in S such that {x, y, z} forms a 3-AP with difference d.
    
    # Let's use a Greedy Construction based on "Potential"
    # Initialize with a random set or a dense block, then refine.
    
    # Strategy:
    # 1. Start with a "core" structure that is known to be good (e.g., 1..32) 
    #    or a random set.
    # 2. Use Simulated Annealing.
    #    - State: bitset b
    #    - Move: Swap one element in S with one element not in S.
    #    - Energy: -score (we want to minimize energy)
    #    - Accept if new_score > old_score or with probability exp((new-old)/T)
    
    # To speed up scoring, we can cache the current score and only recompute 
    # differences? No, bitset AND is very fast in Python. 
    # score_bitset takes ~10-20us. 150s / 20us = 7.5M iterations. This is plenty.
    
    # Let's try a more intelligent initialization:
    # A set like {1, 2, 4, 8, ...} (powers of 2) has few 3-APs.
    # A set like {1..32} has many small d.
    # A set spread out might have large d.
    # The optimal set likely mixes small and large differences.
    
    # Let's run multiple SA runs with different initializations.
    
    best_overall_score = 0
    best_overall_bit = 0
    best_overall_set = []
    
    def simulated_annealing(init_bit, time_budget):
        current_bit = init_bit
        current_score = score_bitset(current_bit)
        best_bit = current_bit
        best_score = current_score
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        # Precompute list of all numbers 1..512
        all_nums = list(range(1, N + 1))
        
        while time.time() < time_budget:
            # Pick a random element to remove
            # We need the list of set elements for quick access
            # Converting bitset to list is O(N) = 512. 
            # Doing this every iteration is slow. 
            # Let's maintain a list of elements in the set.
            
            # Actually, for SA, maintaining a list is good.
            # Let's restart SA with a list-based state for efficiency in moves.
            pass
            
    # Let's implement SA with list state for O(1) swap logic (mostly)
    # and fast score calculation.
    
    def sa_run(init_list, time_budget):
        current_set = set(init_list)
        current_list = list(init_list)
        current_bit = set_to_bitset(current_list)
        current_score = score_bitset(current_bit)
        
        best_set = set(current_list)
        best_bit = current_bit
        best_score = current_score
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        step = 0
        
        all_nums = list(range(1, N + 1))
        
        while time.time() < time_budget and T > T_min:
            # Pick random element in set to remove
            idx = random.randint(0, K - 1)
            old_elem = current_list[idx]
            
            # Pick random element NOT in set to add
            # Efficient way: pick random from 1..512, if in set, retry.
            # Since set size is 32/512 ~ 6%, retry is fast.
            while True:
                new_elem = random.randint(1, N)
                if new_elem not in current_set:
                    break
            
            # Perform swap
            current_set.remove(old_elem)
            current_set.add(new_elem)
            current_list[idx] = new_elem
            
            new_bit = current_bit & ~(1 << (old_elem - 1))
            new_bit |= (1 << (new_elem - 1))
            
            new_score = score_bitset(new_bit)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_bit = new_bit
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_bit = new_bit
                    best_set = set(current_set)
            else:
                # Revert
                current_set.remove(new_elem)
                current_set.add(old_elem)
                current_list[idx] = old_elem
                # current_bit and current_score remain unchanged (they were not updated)
            
            T *= cooling
            step += 1
            
        return best_set, best_bit, best_score

    # Initializations
    inits = []
    
    # 1. Dense block
    inits.append(list(range(1, 33)))
    
    # 2. Random
    inits.append(random.sample(range(1, N + 1), K))
    
    # 3. "Sparse" - spread out
    # Pick 32 numbers roughly evenly spaced
    step = N // K
    spread = [1 + i * step for i in range(K)]
    inits.append(spread)
    
    # 4. "Mixed" - half dense, half random
    half_dense = list(range(1, 17))
    half_random = random.sample(range(33, N + 1), 16)
    inits.append(half_dense + half_random)
    
    # 5. Another random
    inits.append(random.sample(range(1, N + 1), K))
    
    # Run SA for each init with a time slice
    num_runs = len(inits)
    time_per_run = (deadline - time.time() - 10.0) / num_runs
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        run_deadline = time.time() + time_per_run
        # Ensure at least 1s per run
        run_deadline = max(run_deadline, time.time() + 1.0)
        
        best_set, best_bit, best_score = sa_run(init, run_deadline)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_bit = best_bit
            best_overall_set = list(best_set)
    
    # Final Polish: Run SA on the best result found so far with remaining time
    if best_overall_set and time.time() < deadline - 5.0:
        final_deadline = time.time() + (deadline - time.time() - 5.0)
        final_set, final_bit, final_score = sa_run(best_overall_set, final_deadline)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_bit = final_bit
            best_overall_set = list(final_set)
    
    # Ensure valid output
    result = sorted(best_overall_set) if best_overall_set else list(range(1, 33))
    
    # Validation
    if len(result) != K:
        result = list(range(1, 33))
    if len(set(result)) != K:
        result = list(range(1, 33))
    if not all(1 <= x <= N for x in result):
        result = list(range(1, 33))
        
    return result

# EVOLVE-BLOCK-END
