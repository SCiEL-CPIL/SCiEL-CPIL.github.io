import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Parameters are fixed by the problem description effectively, 
    # but we should handle the general case or at least the specific one robustly.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the task description specifies 8 matches, ternary, radius 1.
    # I will implement it generally for q=3, n=8, r=1 but allow parameters.
    
    if q != 3 or n != 8 or radius != 1:
         # Fallback to a simple construction for other cases to avoid errors, 
         # though the evaluator likely only calls with 3, 8, 1.
         # Direct sum for general q, n, r=1?
         # For now, return the seed solution if params don't match, 
         # or a simple greedy if they do.
         # Given the specific task contract, it's 3, 8, 1.
         pass

    N = q ** n
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Precompute neighbors
    # List of all words
    # Convert 0..N-1 to base q digits?
    # It's faster to generate neighbors on the fly or precompute once.
    
    # Precompute neighbors for all N words
    # Shape: (N, 1 + 2*n)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        
        # Get digits
        temp = w
        digits = [0] * n
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            d = digits[i]
            # change digit i to (d+1)%q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            # change digit i to (d+2)%q
            w2 = w + ((d + 2) % q - d) * (q ** i)
            
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1
            
    # Seed solution for comparison
    # Tetracode construction:
    # h = [a, b, (a+b)%3, (a+2b)%3]
    # Code = { h + tail | h in Tetracode, tail in {0,1,2}^4 }
    # This gives 9 * 3^4 = 9 * 81 = 729? No.
    # Tetracode has 9 codewords.
    # Product with 3^4 tails gives 9 * 81 = 729.
    # Wait, the known bound is 486.
    # The Tetracode is a (4, 9, 2) code? No, Tetracode is a ternary code of length 4, size 9, distance 3?
    # Let's check distance.
    # 0000 vs 1112?
    # Actually, the standard construction for K_3(8,1) <= 486 uses the Tetracode.
    # Tetracode is a (4, 9, 3) code?
    # If d=3, then covering radius of Tetracode is 1?
    # If we take product of Tetracode and {0,1,2}^4, the covering radius is 1+0=1?
    # Size = 9 * 3^4 = 729.
    # 729 is larger than 486.
    # So the Tetracode product is NOT the 486 construction.
    # The 486 construction is likely different.
    # However, 729 is a valid upper bound.
    # The current program uses this as a seed.
    # Let's stick to a valid seed.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            # Convert to int
            val = 0
            for i, d in enumerate(word):
                val += d * (q ** i)
            seed_code_ints.append(val)
            
    seed_size = len(seed_code_ints)
    best_code = seed_code_ints
    best_size = seed_size
    
    # Initialize coverage counts for the seed
    cov_counts = np.zeros(N, dtype=np.int32)
    for c in best_code:
        cov_counts[neighbors[c]] += 1
        
    # Verify seed is valid
    if np.min(cov_counts) < 1:
        # Fallback to greedy if seed is invalid (should not happen)
        # But let's assume it is valid.
        pass

    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    # We want to minimize the number of codewords.
    # State: set of codewords, coverage counts.
    # Move: Remove one codeword c1, add one codeword c2.
    # Delta size = 0 (if we keep size constant) or we can allow size changes?
    # Usually, to reduce size, we need a "remove" move that is valid, or a "swap" that leads to a removal later.
    # A common approach:
    # 1. Start with valid code.
    # 2. Try to remove a codeword. If valid, accept.
    # 3. If not, try to swap (remove c1, add c2). Accept based on SA criterion.
    #    The "energy" is the number of uncovered words (should be 0) + lambda * size.
    #    But we must maintain validity (uncovered=0).
    #    If we allow temporary invalidity, we can escape local minima.
    
    # Let's use a metric: E = num_uncovered + 100 * len(code)
    # We want to minimize E.
    # Start with valid code, so num_uncovered=0.
    # If we remove c1 and it's redundant, len decreases, E decreases. Accept.
    # If we remove c1 and it's NOT redundant, num_uncovered increases.
    #    If we then add c2 that covers them, num_uncovered might go back to 0.
    #    This is a 2-step move.
    
    # Alternative: Single step moves.
    # 1. Remove c. If valid, accept.
    # 2. Add c. (Increases size, increases E, rarely accepted unless it helps future removals? No, adding doesn't help removals directly unless it allows removing two others? No.)
    # 3. Swap c1 -> c2.
    #    Remove c1, Add c2.
    #    If valid after swap, accept if size is same (neutral) or if it enables further reduction?
    #    Actually, if we just swap, size is constant.
    #    We need to find a sequence of swaps that leads to a state where a removal is possible.
    
    # Better SA Objective:
    # We track `uncovered_count` and `code_size`.
    # Cost = uncovered_count * 1000 + code_size.
    # Start: Cost = code_size.
    # Move: Swap c1, c2.
    #   Temporarily remove c1.
    #   If c1 was essential, some words become uncovered.
    #   Add c2.
    #   If c2 covers the newly uncovered words, uncovered_count might drop to 0.
    #   If uncovered_count == 0, we have a valid code of same size.
    #   This doesn't reduce size directly.
    #   
    # How to reduce size?
    # We need to find a state where a codeword is redundant.
    # SA should explore the space of valid codes of size K to find one with a redundant codeword?
    # Or allow invalid states?
    
    # Let's try:
    # State: Code (list), CovCounts (array).
    # Moves:
    # 1. Remove random c.
    #    - If valid (all cov_counts > 0 for neighbors of c? No, min(cov_counts[neighbors[c]]) > 1):
    #      Accept. Size decreases.
    #    - If invalid:
    #      Reject (or accept with prob based on penalty).
    # 2. Swap c1, c2.
    #    - Remove c1, Add c2.
    #    - Calculate new uncovered.
    #    - If valid: Accept (size same).
    #    - If invalid: Accept with prob exp(-delta_E / T).
    
    # To make progress, we need to bias towards finding redundant nodes.
    
    # Let's implement a simpler "Greedy + Local Search" first, then add SA if needed.
    # Actually, the prompt asks for a DIFFERENT strategy.
    # Current: Greedy Set Cover + Deletion.
    # New: Simulated Annealing on Swap/Remove moves.
    
    code = best_code.copy()
    cov_counts = cov_counts.copy()
    current_size = len(code)
    
    # Precompute list of codewords in code for fast random access
    # We need to map index in 'code' list to the word.
    
    # To speed up swap evaluation:
    # When removing c1:
    #   For each neighbor v of c1:
    #     cov_counts[v] -= 1
    #     if cov_counts[v] == 0: uncovered_increased
    # When adding c2:
    #   For each neighbor v of c2:
    #     if cov_counts[v] == 0: uncovered_decreased
    #     cov_counts[v] += 1
    
    # Let's run SA.
    T = 1.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # We need a way to pick c1 and c2.
    # c1: random from code.
    # c2: random from all N words? Or from a candidate set?
    # Random from N is 6561, which is small.
    
    # Optimization:
    # We can precompute neighbors for all words.
    # We have `neighbors` array.
    
    # Let's run multiple restarts?
    # Or just one long SA run?
    # Given 115s, we can do many steps.
    
    # Step count:
    # Each step involves O(n) operations for update.
    # 100k steps should be fast.
    # Let's aim for ~1M steps if possible.
    
    # To improve efficiency, we can try to remove first.
    # If removal is possible, do it.
    # If not, do a swap.
    
    # Let's track best_size and best_code.
    
    # Initial state
    # code is a list of ints.
    # We need to be able to remove by value.
    # Using a list, removal is O(K). K ~ 500.
    # 1M steps * 500 = 500M ops. Might be slow in Python.
    # Better to use a set? Or numpy array?
    # If we use a set, we lose order, but that's fine.
    # But we need to pick random element from set.
    # Python set doesn't have O(1) random access.
    # We can maintain a list and swap-remove.
    
    # Let's use a list `code_list` and maintain it.
    # To remove an element at index i:
    #   swap with last, pop.
    #   O(1).
    # To add:
    #   append.
    #   O(1).
    
    code_list = code.copy()
    
    # We need to know if a word is in code_list for quick check?
    # Not strictly necessary for the move logic if we pick c1 from code_list and c2 from all words.
    # But if c2 is already in code_list, the swap is a no-op (or removal if we handle it).
    # We should ensure c2 is NOT in code_list for a valid swap move (otherwise it's just a removal).
    # Or we can allow c2 to be in code_list, in which case we just remove c1 and don't add c2 (redundant add).
    # Actually, if c2 is in code_list, adding it again is invalid (distinct codewords).
    # So we must check if c2 is in code.
    # Checking membership in list is O(K).
    # We can use a set `code_set` for O(1) membership, and update it in parallel.
    
    code_set = set(code_list)
    
    for step in range(2000000): # Max steps
        if time.time() > deadline:
            break
            
        # 1. Try to remove a random codeword
        if len(code_list) > 0:
            idx1 = rng.integers(0, len(code_list))
            c1 = code_list[idx1]
            
            # Check if c1 is redundant
            # min(cov_counts[neighbors[c1]]) > 1
            if np.min(cov_counts[neighbors[c1]]) > 1:
                # Redundant, remove it
                # Update cov_counts
                cov_counts[neighbors[c1]] -= 1
                
                # Remove from list and set
                last_idx = len(code_list) - 1
                code_list[idx1] = code_list[last_idx]
                code_list.pop()
                code_set.remove(c1)
                
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_code = code_list.copy()
                
                # Reset temperature? Or just continue.
                # Since we found an improvement, we can restart cooling or just keep going.
                # Let's keep going.
                continue # Next iteration, try again
                
        # 2. If no removal found (or tried and failed), do a swap
        # Pick c1 randomly from code_list
        idx1 = rng.integers(0, len(code_list))
        c1 = code_list[idx1]
        
        # Pick c2 randomly from all words
        c2 = rng.integers(0, N)
        
        # If c2 is already in code, this is not a valid swap (it would be a removal of c1 and no add, or invalid).
        # If c2 == c1, no-op.
        # If c2 in code_set, skip or treat as removal?
        # Let's just skip if c2 in code_set.
        if c2 in code_set:
            continue
            
        # Evaluate Swap
        # Remove c1
        # Count how many neighbors of c1 become uncovered (cov_counts becomes 0)
        # Add c2
        # Count how many neighbors of c2 are currently uncovered (cov_counts is 0) and get covered.
        
        # Note: We must be careful with overlaps.
        # Let's just simulate the change on a copy? No, too slow.
        # Let's track:
        # delta_uncovered = 0
        
        # Step 1: Remove c1
        # For each v in neighbors[c1]:
        #   if cov_counts[v] == 1:
        #     delta_uncovered += 1 (this v becomes uncovered)
        #   cov_counts[v] -= 1 (conceptually)
        
        # Step 2: Add c2
        # For each v in neighbors[c2]:
        #   if cov_counts[v] == 0 (conceptually, after removal):
        #     delta_uncovered -= 1 (this v becomes covered)
        #   cov_counts[v] += 1 (conceptually)
        
        # But we need to know the "conceptual" cov_counts.
        # cov_counts[v]_new = cov_counts[v]_old - 1(if v in N(c1)) + 1(if v in N(c2))
        
        # Let's compute delta_uncovered directly.
        # Uncovered words are those with count 0.
        # A word v becomes uncovered if old_count == 1 and it is in N(c1) but NOT in N(c2) (or in N(c2) but count goes 1->0->1? No.
        # If v in N(c1) and v in N(c2):
        #   old=1 -> 0 (remove) -> 1 (add). Net change 0.
        # If v in N(c1) and not in N(c2):
        #   old=1 -> 0. Becomes uncovered.
        # If v not in N(c1) and v in N(c2):
        #   old=0 -> 1. Becomes covered (if it was uncovered).
        #   old=1 -> 2. No change in covered status.
        
        # So:
        # delta_uncovered = (# v in N(c1) \ N(c2) with old_count==1) - (# v in N(c2) \ N(c1) with old_count==0)
        
        # Let's compute this.
        # N(c1) and N(c2) are small (17 words).
        # We can iterate over N(c1).
        
        # Optimization:
        # Precompute boolean masks? No, N is 6561, 17 bits per word.
        # Just iterate.
        
        n_c1 = neighbors[c1]
        n_c2 = neighbors[c2]
        
        # Create a set of n_c2 for O(1) lookup?
        # n_c2 has 17 elements. Set creation is fast.
        set_n_c2 = set(n_c2)
        
        delta_uncov = 0
        
        # Iterate over n_c1
        for v in n_c1:
            if v not in set_n_c2:
                if cov_counts[v] == 1:
                    delta_uncov += 1
            # else: v is in both, count goes 1->0->1, no change in uncovered status
            
        # Iterate over n_c2
        for v in n_c2:
            if v not in set(n_c1): # We need set_n_c1?
                # We can just check if v in n_c1? n_c1 is array.
                # Better to make set_n_c1.
                pass
        
        # Let's make set_n_c1 too.
        set_n_c1 = set(n_c1)
        
        # Re-evaluate with sets
        delta_uncov = 0
        for v in n_c1:
            if v not in set_n_c2:
                if cov_counts[v] == 1:
                    delta_uncov += 1
                    
        for v in n_c2:
            if v not in set_n_c1:
                if cov_counts[v] == 0:
                    delta_uncov -= 1
                    
        # Current uncovered count is 0 (we maintain valid state or close to it).
        # Actually, if we allow invalid states, we need to track `current_uncovered`.
        # Let's stick to valid states only for simplicity first?
        # If we only allow valid states, delta_uncov must be 0.
        # If delta_uncov > 0, the swap is invalid.
        # If delta_uncov < 0, it means we covered some words that were already covered?
        # No, if state is valid (all counts >= 1), then cov_counts[v] can never be 0.
        # So the second loop `if cov_counts[v] == 0` will never trigger.
        # So delta_uncov is always >= 0.
        # If delta_uncov == 0, the swap is valid.
        # If delta_uncov > 0, the swap is invalid.
        
        # If we restrict to valid swaps, we can't escape local minima where no valid swap exists.
        # But if no valid swap exists, it means for any c1, c2, removing c1 uncovers some word not covered by c2.
        # This implies a strong dependency.
        
        # Let's allow invalid states to escape.
        # We need to track `current_uncovered`.
        # Initialize to 0.
        
        # For now, let's just accept valid swaps (delta_uncov == 0).
        # This is a random walk on the space of valid codes of fixed size.
        # It might help find a configuration where a removal is possible.
        
        if delta_uncov == 0:
            # Perform the swap
            # Update cov_counts
            cov_counts[n_c1] -= 1
            cov_counts[n_c2] += 1
            
            # Update list and set
            code_list[idx1] = c2
            code_set.remove(c1)
            code_set.add(c2)
            
            # Note: current_size remains same.
            # We don't update best_size here.
            
        else:
            # Invalid swap.
            # We could accept with probability exp(-delta_uncov / T).
            # But we need to update state if accepted.
            # Let's skip for now to keep it simple and fast.
            # If we get stuck, we can restart.
            pass
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
    # After SA, do a final greedy removal pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_code = []
        for c in code_list:
            if np.min(cov_counts[neighbors[c]]) > 1:
                cov_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_code.append(c)
        if changed:
            code_list = new_code
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = code_list.copy()

    # Convert integer codewords back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        # Order: digits[0] is LSB.
        # The output list should correspond to coordinates.
        # If coordinate i corresponds to q^i, then word[i] is the digit.
        # My conversion: word.append(temp % q) gives LSB first.
        # So word[0] is coord 0.
        final_code.append(word)
        
    return final_code
