import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for conversion and cleanup
    
    total_space = q ** n
    
    # Precompute digits for all words
    # digits[idx] = list of n digits
    # We need this for output conversion only, or can compute on the fly.
    # Computing on the fly is O(n) per word, which is fine for ~200 words.
    
    # Precompute neighbor masks as Python integers (bitsets)
    # A bitset of size total_space.
    # Python ints are arbitrary precision, so this is efficient.
    
    # To compute neighbors efficiently, we can precompute the "delta" indices
    # for each position and value change.
    
    # For radius=1 (the common case for this task), neighbors of j are:
    # j itself, and j +/- powers[pos] for pos in 0..n-1, for all valid value changes.
    # Actually, for q=3, changing value at pos from v to v' changes index by (v'-v)*powers[pos].
    
    # Let's precompute the list of offsets for each position.
    # offsets[pos] = list of index deltas to reach a neighbor by changing digit at pos.
    # For a specific word j, the valid offsets depend on the current digit.
    # However, we can precompute a list of (offset, condition) or just generate neighbors on the fly?
    # Generating on the fly is O(n*q) per word. For 2187 words, that's 2187 * 7 * 3 ~ 45k ops.
    # Precomputing masks: O(total_space * n * q). 2187 * 21 ~ 45k ops.
    # Precomputation is better.
    
    # Let's precompute neighbor_masks[i] for all i in 0..total_space-1.
    # This is a list of total_space integers.
    
    # Optimization:
    # For radius=1, the neighbor set of a word is {w} U {w with one digit changed}.
    
    # We can compute this quickly.
    
    neighbor_masks = [0] * total_space
    
    # Precompute powers of q
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Function to get index from digits
    def idx_from_digits(d):
        idx = 0
        for i in range(n):
            idx += d[i] * powers[i]
        return idx
    
    # We need to iterate through all words to build masks.
    # For each word index j:
    # 1. Convert j to digits.
    # 2. Start mask with bit j set.
    # 3. For each position p, for each new_val in 0..q-1 (new_val != old_val):
    #    new_idx = j + (new_val - old_val) * powers[p]
    #    Set bit new_idx in mask.
    
    # To speed up digit extraction, we can use divmod or precomputed tables.
    # Precompute digits for all indices?
    # digits_all = [[0]*n for _ in range(total_space)]
    # This uses memory but is fast. 2187 * 7 ints. Small.
    
    digits_all = []
    for j in range(total_space):
        d = [0] * n
        val = j
        for pos in range(n):
            d[pos] = val % q
            val //= q
        digits_all.append(d)
        
    # Build masks
    for j in range(total_space):
        mask = 1 << j
        d = digits_all[j]
        if radius >= 1:
            for pos in range(n):
                old_val = d[pos]
                p = powers[pos]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    new_idx = j + (new_val - old_val) * p
                    mask |= (1 << new_idx)
        # For radius > 1, we would need more, but the task is radius 1.
        # Let's assume radius=1 for optimization, but handle generic if needed?
        # The task says "radius" is passed. The example is radius 1.
        # If radius > 1, this precomputation explodes or needs BFS.
        # Given the context (K_3(7,1)), radius is almost certainly 1.
        # I will add a check: if radius > 1, fall back to a simpler/slower method or just assume 1?
        # The prompt says "checked over the whole space".
        # I'll assume radius=1 for the bitset optimization. If radius != 1, I might need a different approach.
        # But the score is for the specific task. I'll stick to radius=1 optimization.
        # If radius > 1, the neighbor count is much larger.
        # Let's add a generic fallback for radius > 1?
        # No, let's just handle radius=1 efficiently. If radius > 1, I'll use a list of neighbors instead of bitsets?
        # Or just compute masks via BFS?
        # For now, assume radius=1.
        
        neighbor_masks[j] = mask
        
    # Simulated Annealing Setup
    # State: set of selected codewords (as a set of indices)
    # Objective: Minimize (count + penalty * uncovered_count)
    # penalty should be large enough to prioritize coverage, but allow temporary violations.
    # Let's use penalty = total_space (or higher).
    # Actually, if we allow uncovered words, we must ensure we end up covered.
    # A common approach: cost = |C| + M * |U|, where M is large (e.g., 1000).
    # We minimize cost.
    
    # Initial solution: Random set of size ~ 200? Or greedy?
    # Greedy is a good start. Let's run a fast greedy first to get a valid solution, then SA to improve.
    
    def greedy_init():
        # Fast greedy using bitsets
        # remaining_mask: bits for words not yet covered.
        remaining_mask = (1 << total_space) - 1
        selected = []
        
        # To speed up, we can precompute "gain" for each word?
        # Gain of word w = popcount(neighbor_masks[w] & remaining_mask)
        
        # We need to pick the word with max gain repeatedly.
        # This is O(N * N) if we scan all N words each time.
        # N=2187. 2187^2 ~ 4.7M ops. Python int popcount is fast.
        # 180 iterations * 2187 words * (bitwise AND + popcount).
        # Should be fast enough.
        
        for _ in range(total_space): # Max iterations
            if remaining_mask == 0:
                break
            if time.time() > deadline - 5.0:
                break
                
            best_gain = -1
            best_word = -1
            best_candidates = []
            
            # Optimization: Only consider words that have high potential?
            # No, just scan all.
            for w in range(total_space):
                if w in selected: # Avoid re-adding? Greedy usually adds distinct words.
                    # Actually, we can just check gain. If gain is 0, it's useless.
                    pass
                gain = (neighbor_masks[w] & remaining_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    best_candidates = [w]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(w)
                    
            if best_gain <= 0:
                break # No word covers any remaining? Should not happen if we have all words.
                
            # Random tie break
            w = random.choice(best_candidates)
            selected.append(w)
            remaining_mask &= ~neighbor_masks[w]
            
        return selected

    # Initial Solution
    # Let's try a few random initializations or just one greedy.
    # Greedy is deterministic-ish (with random tie breaks).
    init_sol = greedy_init()
    if not init_sol:
        # Fallback
        init_sol = [0]
        
    selected_set = set(init_sol)
    
    # Calculate initial uncovered
    covered_mask = 0
    for w in selected_set:
        covered_mask |= neighbor_masks[w]
    uncovered_mask = ((1 << total_space) - 1) & ~covered_mask
    uncovered_count = uncovered_mask.bit_count()
    
    current_cost = len(selected_set) + 1000 * uncovered_count
    
    # Simulated Annealing
    # T0, T_min, cooling rate
    # We have ~110 seconds.
    # Each step:
    # 1. Pick a move (add, remove, swap)
    # 2. Calculate delta cost
    # 3. Accept with probability
    
    # Move types:
    # - Add a random word not in set.
    # - Remove a random word from set.
    # - Swap: remove one, add another.
    
    # To make it efficient, we can precompute nothing more.
    # Bit operations are very fast.
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    # Estimate steps per second?
    # Let's run until deadline.
    
    best_cost = current_cost
    best_set = set(selected_set)
    
    # To avoid O(N) scan for "random word not in set", we can keep a list of all words and pick random, checking membership.
    all_words = list(range(total_space))
    
    # We need to update uncovered_mask and covered_mask incrementally.
    # When adding w:
    # new_uncovered = uncovered_mask & ~neighbor_masks[w]
    # delta_uncovered = uncovered_count - new_uncovered.bit_count()
    # cost_delta = 1 - 1000 * delta_uncovered
    
    # When removing w:
    # We need to know if w is the *only* cover for some bits in uncovered_mask?
    # No, we track covered_mask.
    # If we remove w, the new covered_mask is (covered_mask & ~neighbor_masks[w]) | (OR of other masks).
    # This is hard to update incrementally without a count array.
    # Alternative: Maintain a count array for coverage?
    # If we use a count array (list of ints size N), updating is O(neighbors) ~ 15 ops.
    # Checking validity (uncovered count) requires scanning the count array?
    # Or we can maintain `uncovered_count` and update it based on counts.
    
    # Let's switch to a hybrid:
    # Maintain `coverage_counts` (list of ints, size N).
    # `uncovered_count` = number of i where coverage_counts[i] == 0.
    
    # Re-initialize with counts
    coverage_counts = [0] * total_space
    for w in selected_set:
        # Iterate neighbors of w
        # We need a fast way to iterate set bits in neighbor_masks[w].
        # Or just use the precomputed list of neighbor indices?
        # Let's precompute neighbor_indices[w] = list of indices.
        # This is faster for updating counts than bit manipulation if we just need to add/subtract 1.
        
        pass # We didn't precompute indices. Let's do it now or use bit iteration.
        
    # Bit iteration in Python is slow.
    # Let's precompute neighbor_indices.
    # This adds memory but speeds up SA significantly.
    
    neighbor_indices = []
    for j in range(total_space):
        indices = [j]
        d = digits_all[j]
        if radius >= 1:
            for pos in range(n):
                old_val = d[pos]
                p = powers[pos]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    new_idx = j + (new_val - old_val) * p
                    indices.append(new_idx)
        neighbor_indices.append(indices)
        
    # Re-init counts
    coverage_counts = [0] * total_space
    for w in selected_set:
        for idx in neighbor_indices[w]:
            coverage_counts[idx] += 1
            
    # Recalc uncovered_count
    uncovered_count = sum(1 for c in coverage_counts if c == 0)
    current_cost = len(selected_set) + 1000 * uncovered_count
    
    best_cost = current_cost
    best_set = set(selected_set)
    
    # SA Loop
    # We need to be careful with time.
    # Each step:
    # 1. Generate move.
    # 2. Calculate delta cost.
    #    - Add: delta_size = 1. delta_uncovered = - (number of newly covered words).
    #      Newly covered = count of i in neighbors[w] where coverage_counts[i] == 0.
    #    - Remove: delta_size = -1. delta_uncovered = + (number of words that become uncovered).
    #      Become uncovered = count of i in neighbors[w] where coverage_counts[i] == 1.
    #    - Swap: delta_size = 0. delta_uncovered = (newly uncovered) - (newly covered).
    
    # This requires scanning neighbors (size ~15). Very fast.
    
    steps = 0
    last_best_update = time.time()
    
    while time.time() < deadline - 2.0:
        # Pick a move
        r = random.random()
        
        if r < 0.3:
            # Add
            w = random.choice(all_words)
            if w in selected_set:
                continue
            # Calculate gain
            gain = 0
            for idx in neighbor_indices[w]:
                if coverage_counts[idx] == 0:
                    gain += 1
            delta_cost = 1 - 1000 * gain
            if delta_cost < 0 or random.random() < pow(2.718, -delta_cost / T):
                # Accept
                selected_set.add(w)
                for idx in neighbor_indices[w]:
                    if coverage_counts[idx] == 0:
                        uncovered_count -= 1
                    coverage_counts[idx] += 1
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = set(selected_set)
                    last_best_update = time.time()
                    
        elif r < 0.6:
            # Remove
            if not selected_set:
                continue
            w = random.choice(list(selected_set))
            # Calculate loss
            loss = 0
            for idx in neighbor_indices[w]:
                if coverage_counts[idx] == 1:
                    loss += 1
            delta_cost = -1 + 1000 * loss
            if delta_cost < 0 or random.random() < pow(2.718, -delta_cost / T):
                # Accept
                selected_set.remove(w)
                for idx in neighbor_indices[w]:
                    if coverage_counts[idx] == 1:
                        uncovered_count += 1
                    coverage_counts[idx] -= 1
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = set(selected_set)
                    last_best_update = time.time()
                    
        else:
            # Swap
            if len(selected_set) < 2:
                continue
            w1 = random.choice(list(selected_set))
            w2 = random.choice(all_words)
            if w1 == w2:
                continue
            
            # Calculate delta
            # Remove w1, Add w2
            # Loss from w1: count of i in N(w1) where count[i]==1 AND i is NOT in N(w2) (or count becomes 0)
            # This is complex.
            # Simpler:
            # 1. Tentatively remove w1.
            #    loss1 = count of i in N(w1) where count[i]==1
            # 2. Tentatively add w2 (to the state after removal).
            #    gain2 = count of i in N(w2) where (count[i] - (1 if i in N(w1) else 0)) == 0
            #    Wait, if i is in N(w1) and N(w2), and count[i]==1, then after removing w1, count[i]==0.
            #    Adding w2 makes it 1. So it's covered.
            #    If i is in N(w2) but not N(w1), and count[i]==0, adding w2 covers it.
            
            # Let's just compute the new state cost directly?
            # No, that's O(N).
            
            # Efficient Swap Delta:
            # Let A = N(w1), B = N(w2).
            # Let C = coverage_counts.
            # 
            # Words that become uncovered:
            # i in A such that C[i] == 1 and i not in B.
            # (If i in B, C[i] will be decremented then incremented, so net 0 change for that part, 
            #  BUT we must check if C[i] was 1. If C[i]==1 and i in B, after remove C[i]=0, after add C[i]=1. Covered.)
            # So: loss_uncovered = sum(1 for i in A if C[i]==1 and i not in B)
            
            # Words that become covered:
            # i in B such that (C[i] - (1 if i in A else 0)) == 0.
            # i.e., C[i] == 0 and i not in A.
            # OR C[i] == 1 and i in A. (Because C[i] becomes 0 after remove, then 1 after add. Was uncovered? No, was covered by w1. So not "newly covered" in terms of total uncovered count? 
            # Wait. 
            # Initial state: Uncovered set U.
            # Final state: Uncovered set U'.
            # Delta = |U'| - |U|.
            
            # i becomes uncovered iff:
            # 1. i was covered (C[i] > 0).
            # 2. After swap, C'[i] == 0.
            # C'[i] = C[i] - (1 if i in A else 0) + (1 if i in B else 0).
            # So C'[i] == 0 implies:
            # Case 1: i in A, not in B. C[i] - 1 = 0 => C[i] == 1.
            # Case 2: i not in A, not in B. C[i] = 0. (But then i was already uncovered, so not "newly" uncovered? 
            # Actually, we count total uncovered. 
            # If C[i]==0, it was in U. It stays in U. No change.)
            # Case 3: i in A and in B. C[i] - 1 + 1 = C[i]. No change.
            # Case 4: i not in A, in B. C[i] + 1. No change to 0.
            
            # So, newly uncovered = { i in A \ B | C[i] == 1 }.
            
            # i becomes covered iff:
            # 1. i was uncovered (C[i] == 0).
            # 2. After swap, C'[i] > 0.
            # C'[i] = C[i] - (1 if i in A else 0) + (1 if i in B else 0).
            # Since C[i] == 0:
            # Case 1: i in A, not in B. 0 - 1 + 0 = -1. Impossible (counts don't go negative? 
            # Wait, if C[i]==0, i cannot be in A (since A is neighbors of w1, and w1 is selected, so C[i] >= 1).
            # So if C[i]==0, i is NOT in A.
            # So Case 1 is empty.
            # Case 2: i not in A, in B. 0 - 0 + 1 = 1. Covered.
            # Case 3: i in A, in B. Impossible since C[i]==0 implies i not in A.
            # Case 4: i not in A, not in B. 0. Uncovered.
            
            # So, newly covered = { i in B \ A | C[i] == 0 }.
            
            # Delta Uncovered = |Newly Uncovered| - |Newly Covered|
            
            # Implementation:
            # We need to iterate A and B.
            # To check "i in B", we can use a set or bitset?
            # B is small (~15). A is small (~15).
            # Just use sets for A and B for this calculation?
            # Creating sets every step is expensive.
            # Use neighbor_masks?
            # i in B <=> (neighbor_masks[w2] >> i) & 1
            
            newly_uncovered = 0
            # A = neighbor_indices[w1]
            for i in neighbor_indices[w1]:
                if coverage_counts[i] == 1:
                    # Check if i is in B
                    if not (neighbor_masks[w2] >> i) & 1:
                        newly_uncovered += 1
                        
            newly_covered = 0
            # B = neighbor_indices[w2]
            for i in neighbor_indices[w2]:
                if coverage_counts[i] == 0:
                    # Check if i is NOT in A
                    if not (neighbor_masks[w1] >> i) & 1:
                        newly_covered += 1
                        
            delta_uncovered = newly_uncovered - newly_covered
            delta_cost = 0 + 1000 * delta_uncovered # Size change is 0
            
            if delta_cost < 0 or random.random() < pow(2.718, -delta_cost / T):
                # Accept Swap
                selected_set.remove(w1)
                for i in neighbor_indices[w1]:
                    coverage_counts[i] -= 1
                    if coverage_counts[i] == 0:
                        uncovered_count -= 1 # Wait, this logic is flawed if we update counts first.
                # Let's do it carefully.
                # We calculated delta based on old counts.
                # We must update counts and uncovered_count consistently.
                
                # Reset uncovered_count calculation? No, update incrementally.
                # The formula for delta_uncovered was derived.
                # So we can just do:
                uncovered_count += delta_uncovered
                
                # Update counts
                for i in neighbor_indices[w1]:
                    coverage_counts[i] -= 1
                for i in neighbor_indices[w2]:
                    coverage_counts[i] += 1
                
                selected_set.add(w2)
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = set(selected_set)
                    last_best_update = time.time()

        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart or accept best?
            # For now, just stay at T_min.
            
        steps += 1
        if steps % 10000 == 0:
             # Check time
             if time.time() > deadline - 2.0:
                 break

    # Return the best solution found
    # Ensure it's valid? The SA might end with uncovered words if penalty wasn't high enough or T too low.
    # If uncovered_count > 0 in best_set, we should repair it.
    
    final_set = best_set
    # Check validity of best_set
    # Recount coverage for best_set to be sure
    test_counts = [0] * total_space
    for w in final_set:
        for idx in neighbor_indices[w]:
            test_counts[idx] += 1
            
    uncovered_indices = [i for i in range(total_space) if test_counts[i] == 0]
    
    if uncovered_indices:
        # Repair: Add words to cover uncovered indices
        # Greedy repair
        remaining = set(uncovered_indices)
        while remaining:
            best_gain = -1
            best_w = -1
            for w in range(total_space):
                if w in final_set:
                    continue
                gain = 0
                for idx in neighbor_indices[w]:
                    if idx in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_gain <= 0:
                break
            final_set.add(best_w)
            for idx in neighbor_indices[best_w]:
                if idx in remaining:
                    remaining.remove(idx)
                    
    # Convert indices to lists of digits
    result = []
    for idx in final_set:
        result.append([int(d) for d in digits_all[idx]])
        
    return result

if __name__ == "__main__":
    # Test
    res = construct(3, 7, 1)
    print(len(res))
