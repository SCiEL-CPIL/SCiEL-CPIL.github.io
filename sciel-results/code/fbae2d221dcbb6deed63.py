import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    N = 512
    K = 32

    def get_score(m):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in the set represented by bitmask m.
        """
        score = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score

    def get_set_bits(m):
        return [i for i in range(N) if (m >> i) & 1]

    def get_unset_bits(m):
        return [i for i in range(N) if not ((m >> i) & 1)]

    def init_random():
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        return mask

    def init_contiguous():
        mask = 0
        for i in range(K):
            mask |= (1 << i)
        return mask

    def init_two_blocks():
        mask = 0
        for i in range(16):
            mask |= (1 << i)
        for i in range(200, 216):
            mask |= (1 << i)
        return mask

    def init_sparse():
        step = N // K
        mask = 0
        for i in range(K):
            idx = i * step
            mask |= (1 << idx)
        return mask

    def init_arithmetic_progression():
        # Start with a dense arithmetic progression structure
        # e.g., 1, 1+16, 1+32, ... or similar
        # Or a block of 32 consecutive numbers
        mask = 0
        start = random.randint(0, N - K)
        for i in range(K):
            mask |= (1 << (start + i))
        return mask

    initializers = [init_random, init_contiguous, init_two_blocks, init_sparse, init_arithmetic_progression]

    best_mask = 0
    best_score = -1

    # We will run multiple restarts.
    # The previous program used 10 restarts. 
    # Let's try to optimize the SA loop itself to be faster or more effective.
    
    # Key improvement idea: 
    # 1. Use a faster scoring function if possible? The bitwise one is already quite fast.
    # 2. Better move generation? 
    #    - Currently: random swap.
    #    - Alternative: "2-opt" style? Or move one element to a better position?
    #    - Or: Insertion heuristic?
    
    # Let's try a hybrid: SA with random swaps, but also include "re-insertion" moves.
    # Or simply tune the SA parameters.
    
    # Another idea: The greedy polish at the end is expensive (15360 swaps * 256 checks).
    # We can make the polish smarter:
    # Instead of checking ALL swaps, we can check swaps that are likely to help.
    # Or, we can run multiple rounds of greedy polish with different strategies.
    
    # Let's stick to the proven SA + Greedy Polish but refine the Polish.
    # The Polish currently does a full pass. If we find an improvement, it restarts the pass.
    # This can be slow if many improvements are found.
    # We can limit the number of polish iterations or use a time budget more strictly.
    
    # Also, let's add a "hill climbing" phase that tries to insert elements into gaps?
    # No, the set size is fixed.
    
    # Let's try to improve the SA acceptance criteria or temperature schedule.
    # Or, let's try to increase the number of restarts by making each restart faster.
    # If we reduce the SA time per restart, we can do more restarts.
    # But the Polish is often where the big gains come from.
    
    # Let's try a different Polish: 
    # For each element in the set, try moving it to every other position.
    # This is the same as the current swap logic (remove i, add j).
    # The current logic is: for i in set, for j in unset: check swap.
    # This is O(K * (N-K) * CostScore).
    # CostScore is O(N) bitwise ops.
    # 32 * 480 * 256 ~ 4M ops. In Python, this is ~0.1-0.2s.
    # If we do this multiple times, it adds up.
    
    # What if we pre-compute some contributions?
    # Score = sum_{d=1}^{255} I( (m & (m>>d) & (m>>(2d))) != 0 )
    # This is a step function. Hard to pre-compute incrementally for swaps easily without complex bookkeeping.
    # However, we can update the score incrementally if we track which d's are "satisfied".
    # Let's try that.
    
    def get_score_and_satisfied(m):
        """
        Returns (score, satisfied_d_mask) where satisfied_d_mask has bit d set if d is satisfied.
        """
        score = 0
        sat_mask = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
                sat_mask |= (1 << d)
        return score, sat_mask

    def get_score_delta(m, i, j, sat_mask):
        """
        Calculate the change in score and new sat_mask if we swap i (remove) and j (add).
        i is currently set, j is currently unset.
        """
        new_mask = m ^ (1 << i) ^ (1 << j)
        
        # This is O(256) per call.
        # The old get_score is also O(256).
        # So it's not faster.
        # BUT, if we can avoid the full check for d's that are unlikely to change...
        # A swap affects d if i, j, i+d, j+d, i+2d, j+2d are involved.
        # Specifically, d is affected if any of the bits involved in the check for d change.
        # The check for d involves bits a, a+d, a+2d.
        # If we change bit i and bit j, then d is affected if:
        # i is one of {a, a+d, a+2d} for some a, OR j is one of {a, a+d, a+2d} for some a.
        # This means d is in the set of differences related to i or j.
        # For a fixed i, the d's that could be affected are those where i = a, i = a+d, or i = a+2d.
        # i = a => d can be anything? No, a, a+d, a+2d. If i=a, then we need i+d and i+2d to be in set.
        # So d is affected if i is part of a potential triplet.
        # Actually, any d such that i, i+d, i+2d OR i-d, i, i+d OR i-2d, i-d, i are involved.
        # So d is affected if i is in {x, x+d, x+2d} for some x.
        # This implies d is related to i's position.
        # Specifically, if we change i, the status of d depends on neighbors of i at distance d and 2d.
        # So only d's where i-d, i, i+d or i-2d, i-d, i or i, i+d, i+2d are the triplet.
        # This limits the number of d's to check to O(1) per i?
        # No, for a fixed i, the d's that involve i in a triplet are:
        # 1. i is start: d such that i+d, i+2d in S.
        # 2. i is middle: d such that i-d, i+d in S.
        # 3. i is end: d such that i-2d, i-d in S.
        # So for a given i, the potential d's are those where i is start, middle, or end.
        # The number of such d's is at most 255 (if i is small, d can be large).
        # So it's still O(N) in the worst case.
        
        # However, we can check only the d's that were previously satisfied or might become satisfied.
        # If d was satisfied, and we remove i, d might become unsatisfied.
        # If d was unsatisfied, and we add j, d might become satisfied.
        
        # Let's stick to the simple get_score for now, as it's robust.
        # But let's optimize the Polish loop.
        
        # Instead of full restart of Polish on every improvement, 
        # we can do a "best improvement" hill climb:
        # Find the best swap, apply it, repeat.
        # This is what the current code does.
        
        # Let's try to speed up get_score using NumPy?
        # No, Python bitwise is fast.
        
        # Let's try to increase the number of SA steps by reducing the time per step.
        # The main cost is get_score.
        
        # Alternative Strategy: 
        # Use a different representation or algorithm?
        # The problem is small enough that we might be able to do a more exhaustive search.
        # But 512 choose 32 is huge.
        
        # Let's try to improve the initializers.
        # Maybe start with a known good structure?
        # E.g., a set of 32 numbers that forms a grid?
        
        # Let's try to run more restarts with less time each.
        # And improve the Polish to be faster.
        
        # Idea: In Polish, instead of checking all swaps, 
        # check only swaps that involve elements that are "bad" (part of few triplets) 
        # or positions that are "good" (close to many triplets).
        # This is heuristic.
        
        # Let's just stick to the current design but tune parameters.
        # The previous score was 4.8667.
        # The inspiration 1 was 4.8.
        # The current program is likely near the limit of this approach.
        
        # Let's try a "multi-start" with a stronger Polish.
        # Polish: 
        # 1. Find best swap.
        # 2. Apply.
        # 3. Repeat until no improvement.
        # This is already done.
        
        # What if we do "2-swap" Polish? 
        # Remove 2, add 2.
        # This has much larger search space (32*480 * 31*479 / 4 ~ 50M).
        # Too slow.
        
        # Let's try to improve the SA cooling schedule.
        # Or use "adaptive" temperature.
        
        # Let's try to use a "reheat" strategy in SA.
        
        # Actually, let's look at the "differences" strategy mentioned in the directive.
        # "Model the problem as a constraint satisfaction on the set of differences D..."
        # This suggests we should focus on satisfying small d's first.
        
        # Let's try a greedy construction:
        # Start with empty set.
        # For d = 1, 2, 3, ...:
        #   Try to add elements to satisfy d.
        #   But we are limited to 32 elements.
        
        # Alternatively, start with a random set, and then try to "fix" unsatisfied d's.
        
        # Let's try a different Polish:
        # For each unsatisfied d, try to find a swap that satisfies it.
        # This is more targeted.
        
        def polish_targeted(mask, sat_mask):
            """
            Try to satisfy unsatisfied d's by swapping elements.
            """
            # Get unsatisfied d's
            unsat_d = [d for d in range(1, 256) if not (sat_mask >> d) & 1]
            
            # For each unsat d, try to find a swap that makes it satisfied.
            # This is complex.
            
            # Let's stick to the global best swap.
            pass

    # Let's just run the previous algorithm but with a few tweaks:
    # 1. More restarts (15 instead of 10) with slightly less time each.
    # 2. Faster Polish by breaking early if no improvement found in a full pass.
    # 3. Slightly higher initial temperature.
    
    num_restarts = 12
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        
        current_score = get_score(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        T = 15.0
        T_min = 0.05
        cooling_rate = 0.9995  # Slower cooling
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        local_best_score = current_score
        local_best_mask = current_mask
        
        restart_start = time.time()
        sa_time_limit = (deadline - restart_start) * 0.7  # Leave more time for Polish
        
        while T > T_min and (time.time() - restart_start) < sa_time_limit:
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_mask = current_mask
                    
            T *= cooling_rate
            
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
            
        # Polish
        polish_mask = local_best_mask
        polish_score = local_best_score
        polish_set_bits = get_set_bits(polish_mask)
        polish_unset_bits = get_unset_bits(polish_mask)
        
        polish_deadline = deadline - 0.5
        improved = True
        polish_iter = 0
        max_polish_iter = 50
        
        while improved and time.time() < polish_deadline and polish_iter < max_polish_iter:
            improved = False
            best_swap_score = polish_score
            best_i = -1
            best_j = -1
            
            for i in polish_set_bits:
                for j in polish_unset_bits:
                    new_mask = polish_mask ^ (1 << i) ^ (1 << j)
                    new_score = get_score(new_mask)
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_i = i
                        best_j = j
                        improved = True
            
            if improved:
                polish_mask = polish_mask ^ (1 << best_i) ^ (1 << best_j)
                polish_score = best_swap_score
                
                polish_set_bits = get_set_bits(polish_mask)
                polish_unset_bits = get_unset_bits(polish_mask)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask
                polish_iter += 1

    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
