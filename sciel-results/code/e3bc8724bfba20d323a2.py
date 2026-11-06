import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin
    
    N = 1000
    max_rep = 2
    
    # Initial deterministic greedy construction
    # This provides a strong, valid starting point quickly.
    S = []
    sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
    
    for x in range(1, N + 1):
        # Check if x can be added
        # 2*x count
        if sum_counts[2 * x] >= max_rep:
            continue
        
        # x + y for y in S
        # We need to check if any y in S causes a sum s where sum_counts[s] >= max_rep
        # Vectorized check:
        if S:
            S_arr = np.array(S)
            sums = x + S_arr
            # Filter sums that are within bounds (they always are since x,y <= N -> s <= 2N)
            # Check if any count is >= max_rep
            if np.any(sum_counts[sums] >= max_rep):
                continue
        
        # Add x
        S.append(x)
        sum_counts[2 * x] += 1
        if S:
            S_arr = np.array(S[:-1])
            sums = x + S_arr
            sum_counts[sums] += 1
            
    best_set = S[:]
    best_size = len(S)
    
    # Local Search Phase
    # Strategy:
    # 1. Try to add any missing element (expansion).
    # 2. Try 1-in/1-out swaps (replace an element with a missing one to potentially allow further additions or just rearrange).
    # 3. Try 2-in/1-out (remove one, add two) to increase size by 1.
    
    # For efficiency, we maintain a list of candidates and a set of current elements.
    # We will iterate through potential moves.
    
    # To speed up feasibility checks for swaps, we can pre-compute or use vectorized ops.
    # Let's define a helper to check if a set of additions is valid given current state and removals.
    
    # We'll use a simple loop for the local search, but with vectorized inner checks.
    # Since N=1000, O(N^2) per move is 1e6 ops. Vectorized numpy is fast.
    
    # We will try random restarts of the local search or just iterate through all possible swaps?
    # There are ~1000 elements and ~1000 missing. 1e6 swaps. Too many if each is slow.
    # We need to be selective.
    
    # Let's try a hill-climbing approach:
    # Iterate through all possible 1-in/1-out swaps. If it improves (allows more additions later? or just valid?), keep it.
    # Actually, the goal is to MAXIMIZE SIZE.
    # So, a move is only useful if it leads to a larger set.
    # Direct 1-in/1-out doesn't change size. It's only useful if it "unblocks" an addition.
    # So the real move is: Remove A, Add B, then try to add C.
    # This is a 2-step lookahead.
    
    # Alternative: Simulated Annealing or Random Walk?
    # Given the time limit, let's try a simpler "Greedy with Backtracking" or "Local Search with Size Increase".
    
    # Let's try:
    # 1. Find all "blocked" additions (elements that can't be added).
    # 2. For each blocked element X, see if removing one existing element Y allows X to be added.
    #    If so, we have a swap (Y out, X in). Size stays same.
    #    Then, after the swap, check if we can add another element Z.
    #    If yes, we have a net gain of 1 (Y out, X in, Z in).
    
    # This is expensive. Let's simplify.
    # Let's just do random 1-in/1-out swaps to shuffle the set, and periodically try to add new elements.
    # This is similar to the seed but more focused on rearrangement.
    
    # Better:
    # Use the deterministic greedy set as the base.
    # Try to improve it by finding "holes" that can be filled by removing a "useless" element.
    
    # Let's implement a function to check if a specific element can be added.
    def can_add(x, S_list, sum_counts_arr):
        if sum_counts_arr[2 * x] >= max_rep:
            return False
        if not S_list:
            return True
        S_arr = np.array(S_list)
        sums = x + S_arr
        if np.any(sum_counts_arr[sums] >= max_rep):
            return False
        return True

    # Let's try a different structural approach:
    # Start with the greedy set.
    # Perform random swaps to explore the neighborhood.
    # If a swap is valid, accept it.
    # After each swap, try to greedily add any new elements that become possible.
    
    current_S = S[:]
    current_counts = sum_counts.copy()
    
    # Precompute indices for speed
    # We will iterate for a fixed number of steps or until time runs out.
    
    # To make this fast, we can batch the "try to add" step.
    
    # Let's try a specific heuristic:
    # The greedy set is "maximal" in the sense that no single element can be added.
    # To increase size, we MUST remove at least one element.
    # Let's try removing each element one by one, and then greedily filling the space.
    # This is O(N) removals * O(N) greedy fill = O(N^2).
    # N=1000, so 1e6 steps. Each step involves vectorized checks.
    # This might be feasible.
    
    # Algorithm:
    # For each element r in current_S:
    #   1. Remove r from S and counts.
    #   2. Run a greedy pass to add as many new elements as possible (starting from 1..N).
    #   3. If the resulting size is > best_size, update best.
    #   4. Restore r (or just restart from best found).
    
    # This is a "1-removal" local search.
    
    # Optimization:
    # Instead of restarting the greedy pass from scratch, we can just try to add the elements that were previously blocked.
    # But the counts change, so many new elements might be addable.
    
    # Let's implement the "Remove r, then Greedy Fill" approach.
    
    # We need a fast greedy fill function.
    def greedy_fill(S_list, counts_arr, candidates):
        # candidates is a list of elements to try adding
        # We modify S_list and counts_arr in place
        for x in candidates:
            if x in S_list: # This check is O(N). We should use a set or boolean array.
                continue
            # Check validity
            if counts_arr[2 * x] >= max_rep:
                continue
            if S_list:
                S_arr = np.array(S_list)
                sums = x + S_arr
                if np.any(counts_arr[sums] >= max_rep):
                    continue
            
            # Add x
            S_list.append(x)
            counts_arr[2 * x] += 1
            if S_list:
                S_arr = np.array(S_list[:-1])
                sums = x + S_arr
                counts_arr[sums] += 1

    # To make "x in S_list" fast, we maintain a boolean array `in_set`.
    
    # Let's refine the local search loop.
    
    # State:
    # S_list: list of current elements
    # in_set: boolean array of size N+1
    # counts: numpy array of size 2N+1
    
    # We will try removing each element in the current best set and see if we can grow.
    
    # To avoid O(N) membership check, use in_set.
    
    # Let's run this for a few iterations or until time limit.
    
    # Initial state from deterministic greedy
    S_list = best_set[:]
    in_set = np.zeros(N + 1, dtype=bool)
    for x in S_list:
        in_set[x] = True
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    for i in range(len(S_list)):
        x = S_list[i]
        counts[2 * x] += 1
        for j in range(i):
            y = S_list[j]
            counts[x + y] += 1
            
    best_size = len(S_list)
    
    # Local Search: Try removing each element and refilling
    # We will cycle through the elements in S_list.
    
    # To speed up the refill, we only need to check elements that are NOT in the set.
    # And we can stop early if we exceed the time limit.
    
    # We will try a limited number of removals, prioritizing "central" elements or random ones?
    # Let's just iterate through all elements in S_list.
    
    # Since N=1000, and we have ~1000 elements, this is 1000 removals.
    # Each removal triggers a greedy fill.
    # Greedy fill checks ~1000 candidates.
    # Total checks ~ 1e6.
    # Each check is vectorized.
    
    # Let's implement the removal-refill loop.
    
    # We need to be careful with time.
    
    # We will keep a list of "promising" removals? No, just iterate.
    
    # We will store the best set found during this phase.
    
    # To avoid modifying the original best_set while iterating, we work on copies.
    
    # Optimization:
    # Instead of copying the whole array and list for each removal, we can do it in-place and backtrack.
    # But backtracking a greedy fill is complex.
    # Copying 1000 ints is fast. Copying 2000 ints is fast.
    
    # Let's do:
    # for r in S_list:
    #   if time > deadline: break
    #   S_temp = S_list.copy()
    #   counts_temp = counts.copy()
    #   in_set_temp = in_set.copy()
    #   
    #   # Remove r
    #   S_temp.remove(r)
    #   in_set_temp[r] = False
    #   # Update counts for removal
    #   # This is O(N)
    #   for y in S_temp:
    #       counts_temp[r + y] -= 1
    #   counts_temp[2 * r] -= 1
    #   
    #   # Greedy Fill
    #   # We want to add elements. Which ones?
    #   # We can try all elements from 1 to N.
    #   # To be fast, we can just iterate 1..N.
    #   for x in range(1, N + 1):
    #       if in_set_temp[x]:
    #           continue
    #       # Check if x can be added
    #       if counts_temp[2 * x] >= max_rep:
    #           continue
    #       if S_temp:
    #           S_arr = np.array(S_temp)
    #           sums = x + S_arr
    #           if np.any(counts_temp[sums] >= max_rep):
    #               continue
    #       
    #       # Add x
    #       S_temp.append(x)
    #       in_set_temp[x] = True
    #       counts_temp[2 * x] += 1
    #       if S_temp:
    #           S_arr = np.array(S_temp[:-1])
    #           sums = x + S_arr
    #           counts_temp[sums] += 1
    #           
    #   if len(S_temp) > best_size:
    #       best_size = len(S_temp)
    #       best_set = S_temp
    #       # Update the main state to the new best?
    #       # Yes, to allow further improvements from this new base.
    #       S_list = best_set
    #       counts = counts_temp
    #       in_set = in_set_temp
    #       # Restart the loop? Or continue?
    #       # If we update S_list, the `for r in S_list` loop might behave oddly if we modify it.
    #       # We should break and restart the outer loop or handle it carefully.
    #       # Let's just break and restart the local search phase from the new best.
    #       break
    
    # This "remove one, refill" strategy is a good structural change.
    # Let's implement it cleanly.
    
    # We'll wrap it in a while loop that runs until time runs out.
    # Each iteration:
    # 1. Pick a random element to remove (or iterate).
    # 2. Refill.
    # 3. If improved, accept and restart.
    # 4. If not, try next.
    
    # Using random removal might explore the space better than sequential.
    
    # Let's use random removal.
    
    while time.time() < deadline:
        if not S_list:
            break
            
        # Pick a random element to remove
        r_idx = np.random.randint(len(S_list))
        r = S_list[r_idx]
        
        # Copy state
        S_temp = S_list.copy()
        counts_temp = counts.copy()
        in_set_temp = in_set.copy()
        
        # Remove r
        S_temp.remove(r)
        in_set_temp[r] = False
        
        # Update counts for removal
        # We need to subtract 1 from counts[r+y] for all y in S_temp
        # And 1 from counts[2*r]
        if S_temp:
            S_arr = np.array(S_temp)
            sums = r + S_arr
            counts_temp[sums] -= 1
        counts_temp[2 * r] -= 1
        
        # Greedy Fill
        # We iterate through all possible candidates
        # To speed this up, we can shuffle candidates?
        # Deterministic 1..N is fine, but maybe shuffling helps find different local optima?
        # Let's stick to 1..N for speed and determinism in the fill.
        
        # Optimization: Only check candidates that are "close" to the removed element?
        # No, any element might become addable.
        
        for x in range(1, N + 1):
            if in_set_temp[x]:
                continue
            
            # Check validity
            if counts_temp[2 * x] >= max_rep:
                continue
            
            if S_temp:
                S_arr = np.array(S_temp)
                sums = x + S_arr
                if np.any(counts_temp[sums] >= max_rep):
                    continue
            
            # Add x
            S_temp.append(x)
            in_set_temp[x] = True
            counts_temp[2 * x] += 1
            if S_temp:
                S_arr = np.array(S_temp[:-1])
                sums = x + S_arr
                counts_temp[sums] += 1
                
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_set = S_temp
            # Update main state
            S_list = best_set
            counts = counts_temp
            in_set = in_set_temp
            # We can continue the loop, as the new state might allow further improvements
        else:
            # No improvement. We could try a different removal.
            # To avoid getting stuck, we might want to try a different random removal.
            # The while loop will pick a new random one next iteration.
            pass

    return best_set
