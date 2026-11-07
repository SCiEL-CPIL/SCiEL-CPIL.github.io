import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill

    # Strategy:
    # 1. Use NumPy boolean arrays for subset-sum DP.
    #    dp[c] is a boolean array of size MAX_SUM+1 where dp[c][s] is True
    #    if sum s is achievable with exactly c elements.
    #    This is much faster than Python sets for checking disjointness
    #    and for incremental updates.
    # 2. Use Simulated Annealing to find large valid sets.
    #    Moves: Add a random element, Remove a random element, Swap two elements.
    #    Objective: Maximize set size.
    #    Constraint: Validity (disjoint sums across cardinalities).

    MAX_VAL = 256
    MAX_SUM = 256 * 60  # Max possible sum for 60 elements of max 256
    # Actually, we don't need the full MAX_SUM for all cardinalities.
    # But for simplicity and speed, let's use a fixed size.
    # 256 * 60 = 15360. This is small enough for NumPy.
    SUM_SIZE = MAX_SUM + 1

    def is_valid_numpy(arr):
        """
        Check if the set arr is valid using NumPy DP.
        Returns True if valid.
        """
        n = len(arr)
        # dp[c] is a boolean array of length SUM_SIZE
        # dp[0] has 0 at index 0
        dp = [np.zeros(SUM_SIZE, dtype=bool) for _ in range(n + 1)]
        dp[0][0] = True
        
        for x in arr:
            # Update backwards to avoid using x multiple times in same step
            # We need to be careful with in-place updates or use copies.
            # Since we are iterating c from n-1 down to 0:
            # new_dp[c+1] |= old_dp[c] shifted by x
            # We can do this in-place if we go backwards and use a temporary copy
            # or just update dp[c+1] based on dp[c] which hasn't been updated for this x yet.
            
            # Wait, if we do dp[c+1] |= (dp[c] shifted), and we iterate c backwards,
            # dp[c] is still the state from before adding x (for c < current max used).
            # Actually, dp[c] for c < n is not modified by the addition of x to form dp[c+1].
            # The addition of x affects dp[c+1] based on dp[c].
            # So we can update in-place if we go from high c to low c.
            
            for c in range(n - 1, -1, -1):
                if not np.any(dp[c]):
                    continue
                # Shift dp[c] by x
                # dp[c][:-x] becomes the source for dp[c+1][x:]
                # We need to OR this into dp[c+1]
                # But we must ensure we don't use x more than once.
                # Since we are going backwards, dp[c] has not been updated by x yet.
                # So this is safe.
                
                # Efficient shift and OR
                # dp[c+1][x:] |= dp[c][:-x]
                dp[c+1][x:] |= dp[c][:-x]

        # Check disjointness
        # For each c, check if dp[c] has any True where sum has been seen before
        seen = np.zeros(SUM_SIZE, dtype=bool)
        for c in range(n + 1):
            # If there is any overlap between dp[c] and seen, it's invalid
            if np.any(dp[c] & seen):
                return False
            seen |= dp[c]
        return True

    # Helper to check validity of a list of ints
    def check_valid(lst):
        if not lst:
            return True
        return is_valid_numpy(np.array(lst, dtype=np.int32))

    # Initial greedy construction to get a decent starting point
    # Use a fast heuristic: try to add small numbers first?
    # Or just start with a known good set and expand.
    
    # Let's start with a simple set and grow it greedily using the fast checker.
    # Greedy: try to add numbers 1..256 in some order.
    # Order matters. Let's try increasing order.
    current_set = set()
    arr = []
    
    # Fast greedy:
    for x in range(1, 257):
        if x in current_set:
            continue
        new_arr = arr + [x]
        if check_valid(new_arr):
            arr = new_arr
            current_set.add(x)
    
    best_arr = arr.copy()
    best_size = len(best_arr)
    
    # Simulated Annealing
    # State: current set (list)
    # Move: Add, Remove, Swap
    # Objective: Maximize size.
    # If size increases, always accept.
    # If size decreases, accept with probability exp(delta / T).
    # If size stays same (swap), accept with probability exp(delta / T) where delta is 0?
    # Actually, we want to explore.
    
    # Let's define the move space.
    # We can pick a random candidate from 1..256 not in set to add.
    # We can pick a random element from set to remove.
    # We can pick one to remove and one to add (swap).
    
    import random
    rng = random.Random(123)
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99
    
    current_arr = best_arr.copy()
    current_size = len(current_arr)
    current_set = set(current_arr)
    
    # Pre-compute something? No, DP is the bottleneck.
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a move
        move_type = rng.random()
        new_arr = current_arr.copy()
        new_set = current_set.copy()
        
        if move_type < 0.5:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in current_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_arr.append(x)
            new_set.add(x)
            delta_size = 1
        elif move_type < 0.8:
            # Remove a random element
            if not current_arr:
                continue
            idx = rng.randrange(len(current_arr))
            x = current_arr[idx]
            new_arr.pop(idx)
            new_set.remove(x)
            delta_size = -1
        else:
            # Swap: remove one, add another
            if not current_arr:
                continue
            idx = rng.randrange(len(current_arr))
            x_out = current_arr[idx]
            candidates = [x for x in range(1, 257) if x not in current_set]
            if not candidates:
                continue
            x_in = rng.choice(candidates)
            new_arr[idx] = x_in
            new_set.remove(x_out)
            new_set.add(x_in)
            delta_size = 0
            
        # Check validity of new state
        if not check_valid(new_arr):
            # Reject invalid states immediately (or with high penalty)
            # For simplicity, reject.
            continue
            
        new_size = len(new_arr)
        
        # Acceptance criteria
        if new_size > current_size:
            # Always accept improvement
            current_arr = new_arr
            current_set = new_set
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_arr = current_arr.copy()
        else:
            # new_size <= current_size
            # delta = new_size - current_size
            # If delta < 0, accept with prob exp(delta/T)
            # If delta == 0, accept with prob exp(0/T) = 1? 
            # Swaps with same size are useful for escaping local minima.
            # Let's accept swaps with a fixed probability or always?
            # If we always accept swaps, we might cycle.
            # Let's accept swaps with prob 0.5 or similar.
            # Actually, for SA, we usually define an energy function.
            # Here, we want to maximize size.
            # Let Energy = -Size.
            # delta_E = E_new - E_old = -new_size - (-current_size) = current_size - new_size = -delta_size.
            # Accept if rand < exp(-delta_E / T) = exp(delta_size / T).
            
            if delta_size == 0:
                # Swap
                # Accept with some probability to allow exploration
                if rng.random() < 0.5:
                    current_arr = new_arr
                    current_set = new_set
                    # current_size unchanged
            else:
                # delta_size < 0
                prob = np.exp(delta_size / T)
                if rng.random() < prob:
                    current_arr = new_arr
                    current_set = new_set
                    current_size = new_size
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: Restart if stuck?
        # If we haven't improved in a long time, maybe reheat?
        # For now, keep it simple.
        
    return [int(x) for x in best_arr]
