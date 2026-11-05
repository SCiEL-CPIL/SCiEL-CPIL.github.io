import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # This implementation is optimized for q=3, n=8, radius=1
    # which is the specific instance in the task contract.
    # For other parameters, it will still run but might be slower or less effective.
    
    total_words = q ** n
    
    # Precompute digits for all words
    # This is used to convert between index and tuple representation
    # And to generate neighbors efficiently
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists for radius 1
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    # For q=3, n=8, radius=1, each word has 1 + 8*(3-1) = 17 neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                # Calculate index from digits
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
        
    # Convert neighbors to numpy arrays for faster vectorized operations
    # This is a significant speedup for coverage updates
    neighbor_arrays = []
    for i in range(total_words):
        neighbor_arrays.append(np.array(neighbors[i], dtype=np.int32))
        
    # --- Phase 1: Vectorized Greedy Set Cover Initialization ---
    # Start with an empty code
    # We want to cover all total_words points.
    # cov_count[x] = number of codewords covering x
    # uncovered_count = number of x with cov_count[x] == 0
    
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # To make greedy selection efficient, we can just iterate through all words
    # and pick the one that covers the most currently uncovered points.
    # However, doing this from scratch for 6561 words is O(N^2 * K) which might be slow in Python loops.
    # Instead, we can use a randomized greedy approach or a simple heuristic.
    
    # A better initialization: Randomly shuffle all words, then greedily add if it covers new points.
    # This is O(N * K) on average if we are lucky, but worst case is higher.
    # Given N=6561, K=17, N*K is ~111k operations per word. 6561*111k is ~700M. Too slow in pure Python.
    
    # Let's use a smarter heuristic:
    # 1. Pick a random word.
    # 2. Add it.
    # 3. Find the word that maximizes coverage of *uncovered* points.
    #    To do this efficiently, we can precompute for each word the list of neighbors.
    #    When we add a word, we update cov_count.
    #    The "gain" of adding word c is the number of x in neighbors[c] where cov_count[x] == 0.
    
    # We can maintain an array `gain` where gain[c] is the current number of uncovered neighbors of c.
    # Initially, gain[c] = len(neighbors[c]) = 17 for all c.
    # When we add a word w, for each x in neighbors[w]:
    #    if cov_count[x] was 0, it becomes 1.
    #    For every other word c that covers x (i.e., x in neighbors[c]), gain[c] decreases by 1.
    #    This is the expensive part: updating gain for all c covering x.
    #    How many c cover a given x? Exactly 17 (the neighbors of x).
    #    So adding one word w costs sum_{x in neighbors[w]} 17 operations = 17 * 17 = 289 ops.
    #    Total for 486 words: 486 * 289 ~ 140k ops. This is very fast!
    
    # Let's implement this.
    
    gain = np.full(total_words, radius, dtype=np.int32) # Wait, gain is number of uncovered neighbors.
    # Actually, initially all are uncovered. So gain[c] = len(neighbors[c]) = 1 + n*(q-1).
    # Let's call it `potential`.
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    
    # We need to know, for each x, which c cover it.
    # We have neighbors[c]. We can invert this.
    # coverers[x] = list of c such that x in neighbors[c].
    # Note: neighbors is symmetric for radius 1 (Hamming distance).
    # So coverers[x] is just neighbors[x].
    
    # Algorithm:
    # 1. Initialize potential[c] = 17 for all c.
    # 2. Repeat until all covered:
    #    a. Find c with max potential. (Use np.argmax)
    #    b. Add c to code.
    #    c. For each x in neighbors[c]:
    #       if cov_count[x] == 0:
    #           cov_count[x] = 1
    #           # x was uncovered, now covered.
    #           # For all c' covering x, potential[c'] decreases by 1.
    #           # c' are neighbors[x].
    #           potential[neighbors[x]] -= 1
    #       else:
    #           cov_count[x] += 1
    #           # x was already covered. No change to potential?
    #           # Wait, potential[c] is defined as # uncovered neighbors of c.
    #           # If x was already covered, adding c doesn't change the status of x from uncovered to covered.
    #           # So potential[c] for c covering x does NOT decrease.
    #           # Correct.
    
    # One issue: `np.argmax` returns the first maximum. If there are ties, it's deterministic.
    # To introduce randomness and explore different basins, we can break ties randomly.
    # Or simply shuffle the order of candidates with max potential.
    
    # Let's implement this greedy construction.
    
    code_set = set()
    uncovered_remaining = total_words
    
    # We need a fast way to get neighbors[x] as a numpy array for vectorized subtraction.
    # We already have neighbor_arrays.
    
    while uncovered_remaining > 0 and time.time() < deadline - 60.0:
        # Find candidates with max potential
        max_pot = np.max(potential)
        if max_pot == 0:
            break # Should not happen if uncovered_remaining > 0, unless bug
        
        # Get all indices with max potential
        candidates = np.where(potential == max_pot)[0]
        
        # Pick one randomly to break ties and add diversity
        c = random.choice(candidates.tolist())
        
        if c in code_set:
            # This shouldn't happen if we mark them, but just in case
            potential[c] = 0
            continue
            
        code_set.add(c)
        code_list.append(c)
        is_code[c] = True
        
        # Update coverage
        neigh_c = neighbor_arrays[c]
        
        # We need to identify which neighbors were uncovered (cov_count == 0)
        # and which were covered (cov_count > 0).
        
        # Efficient update:
        # 1. Identify uncovered neighbors
        #    mask = cov_count[neigh_c] == 0
        #    uncovered_neigh = neigh_c[mask]
        #    
        # 2. For each x in uncovered_neigh:
        #    cov_count[x] = 1 (was 0)
        #    potential[neighbors[x]] -= 1
        #    
        # 3. For each x in neigh_c[~mask]:
        #    cov_count[x] += 1
        
        # Vectorized approach for step 2 is tricky because we need to update potential
        # for neighbors of neighbors.
        # Let's loop over the 17 neighbors of c.
        
        for x in neigh_c:
            if cov_count[x] == 0:
                # x becomes covered
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Decrease potential for all c' covering x
                # c' are neighbors[x]
                # potential[neighbors[x]] -= 1
                # This is a vectorized operation on a small array (size 17)
                potential[neighbor_arrays[x]] -= 1
            else:
                cov_count[x] += 1
                
        # Ensure the added codeword itself has 0 potential to avoid re-adding
        potential[c] = 0
        
    # Now we have a valid covering code.
    # Size is likely around 500-600.
    
    if not code_list:
        # Fallback if greedy failed (should not happen)
        return [list(digits_of[i]) for i in range(total_words)]
        
    best_code = code_list[:]
    best_size = len(best_code)
    
    # --- Phase 2: Improved Simulated Annealing ---
    # Focus on: better move selection, faster swap eval, aggressive local search
    
    current_code = best_code[:]
    current_code_set = set(current_code)
    current_is_code = is_code.copy()
    current_cov_count = cov_count.copy()
    current_size = len(current_code)
    
    # Precompute neighbor sets for faster membership testing
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # SA Parameters
    T_start = 15.0
    T_end = 0.005
    cooling_rate = 0.998
    T = T_start
    
    # Track best valid code found during SA
    sa_best_size = current_size
    sa_best_code = current_code[:]
    
    # Counter for steps
    step = 0
    steps_per_check = 1000
    
    while time.time() < deadline - 8.0:
        step += 1
        
        # Periodically try aggressive local search
        if step % steps_per_check == 0:
            # Try to remove any possible words
            removed_any = True
            while removed_any and time.time() < deadline - 8.0:
                removed_any = False
                for c1 in current_code[:]:
                    if c1 not in current_code_set:
                        continue
                    removable = True
                    for x in neighbors[c1]:
                        if current_cov_count[x] == 1:
                            removable = False
                            break
                    if removable:
                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        current_code.remove(c1)
                        current_size -= 1
                        removed_any = True
                        if current_size < sa_best_size:
                            sa_best_size = current_size
                            sa_best_code = current_code[:]
        
        # Pick move type: 60% swap, 40% removal
        if random.random() < 0.6 and current_size > 0:
            # Try Swap
            c1 = random.choice(current_code)
            
            # Pick c2 not in code - prefer neighbors of c1 for geometric locality
            c2 = None
            # First try neighbors of c1 (excluding c1 itself)
            for _ in range(10):
                candidate = random.choice(neighbors[c1])
                if candidate != c1 and candidate not in current_code_set:
                    c2 = candidate
                    break
            
            # If no good neighbor found, try random
            if c2 is None:
                for _ in range(15):
                    candidate = random.randint(0, total_words - 1)
                    if candidate not in current_code_set:
                        c2 = candidate
                        break
            
            if c2 is None:
                continue
            
            # Fast swap evaluation using precomputed sets
            n1 = neighbor_sets[c1]
            n2 = neighbor_sets[c2]
            
            # Only check words that might become uncovered: in n1 but not in n2
            # And words that might become covered: in n2 but not in n1 (less important)
            newly_uncovered = 0
            for x in n1:
                if x not in n2:
                    if current_cov_count[x] == 1:
                        newly_uncovered += 1
            
            # Also check words in intersection that have cov_count == 1
            # Actually, if x is in both n1 and n2, cov_count doesn't change
            # So we only need to check n1 \ n2 for uncovered
            
            delta_E = 100 * newly_uncovered
            
            accept = False
            if newly_uncovered == 0:
                accept = True
            else:
                if T > 0.01:
                    if random.random() < np.exp(-delta_E / T):
                        accept = True
            
            if accept:
                # Apply swap
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                
                current_cov_count[neighbors[c2]] += 1
                current_is_code[c2] = True
                current_code_set.add(c2)
                current_code.append(c2)
                
        else:
            # Try Removal
            if current_size == 0:
                continue
                
            c1 = random.choice(current_code)
            
            newly_uncovered = 0
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    newly_uncovered += 1
            
            if newly_uncovered == 0:
                # Valid removal - always accept
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                current_size -= 1
                
                if current_size < sa_best_size:
                    sa_best_size = current_size
                    sa_best_code = current_code[:]
            else:
                delta_E = -1 + 100 * newly_uncovered
                if T > 0.01 and random.random() < np.exp(-delta_E / T):
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code.remove(c1)
                    current_size -= 1
        
        # Cooling
        T = max(T_end, T * cooling_rate)
        
        # Update best if we found a valid smaller code
        if current_size < sa_best_size:
            sa_best_size = current_size
            sa_best_code = current_code[:]
    
    # Use the best code found during SA
    final_code = sa_best_code[:]
    final_code_set = set(final_code)
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    # Aggressive final pruning: multiple passes with different random orders
    for pass_num in range(5):
        if time.time() > deadline - 3.0:
            break
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Create a shuffled copy to try removals in random order
            shuffled = final_code[:]
            random.shuffle(shuffled)
            
            for c1 in shuffled:
                if c1 not in final_code_set:
                    continue
                removable = True
                for x in neighbors[c1]:
                    if final_cov_count[x] == 1:
                        removable = False
                        break
                if removable:
                    final_cov_count[neighbors[c1]] -= 1
                    final_code_set.remove(c1)
                    final_code.remove(c1)
                    improved = True
        
    # Verify final code
    if not np.all(final_cov_count > 0):
        # Fallback to best valid code
        return [list(digits_of[i]) for i in sa_best_code]
        
    return [list(digits_of[i]) for i in final_code]
