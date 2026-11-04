import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute neighbors for each word.
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius.
    # For q=3, n=8, radius=1: 17 neighbors per word.
    
    # We use a flat array representation for speed.
    # digits_of[i] is a tuple of n digits.
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        # Radius 1 neighbors
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                # Calculate index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
    
    # Greedy Set Cover Algorithm
    # State:
    # uncovered: boolean array, True if word is not covered by any selected codeword.
    # selected: set of selected codeword indices.
    # 
    # At each step, find the codeword c that maximizes |neighbors[c] intersect uncovered|.
    # Add c to selected, update uncovered.
    
    # Optimization:
    # Instead of scanning all 6561 words every time (which is 6561 * 17 * 486 operations ~ 54M ops),
    # we can maintain a "gain" count for each potential codeword.
    # gain[c] = number of currently uncovered words in neighbors[c].
    # When we add a codeword c_new, we mark its neighbors as covered.
    # For every other codeword c, if any of its neighbors were just covered, gain[c] decreases.
    # This update is expensive if done naively.
    
    # Alternative: Since the space is small (6561), we can use a heap or a simple loop with numpy vectorization.
    # Let's try a hybrid: 
    # 1. Start with a random or heuristic initial set? No, pure greedy from empty is the strategy.
    # 2. To speed up, we can use NumPy to compute gains for all candidates at once?
    #    Uncovered is a boolean vector of size 6561.
    #    Neighbors can be stored as a matrix? 6561x6561 is too big (43M bools = 43MB, okay).
    #    Or just use the list of neighbors.
    
    # Let's use a more efficient greedy step:
    # We maintain `uncovered` as a numpy boolean array.
    # To find the best candidate, we need to compute the sum of `uncovered[neighbors[c]]` for all c.
    # This is a matrix-vector multiplication if we had a binary incidence matrix.
    # Let's build the incidence matrix `M` where M[c][x] = 1 if dist(c,x) <= radius.
    # M is 6561x6561. 43 million entries. As int8, that's 43MB. Acceptable.
    
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for c in range(total_words):
        M[c, neighbors[c]] = 1
        
    # Greedy Loop
    # We will run multiple restarts with different randomization in tie-breaking or initial perturbations?
    # Pure deterministic greedy might get stuck. 
    # Let's use a "Greedy with Randomized Restart" or "Greedy with Local Improvement".
    
    best_code = []
    best_size = total_words # Max possible
    
    # To allow for different basins, we can slightly randomize the initial "uncovered" state? No, that breaks coverage.
    # Instead, we can use a "Greedy + Prune" hybrid.
    # 1. Run Greedy to get a valid code (might be larger than optimal).
    # 2. Prune it using the deletion logic from the previous strategy.
    
    def run_greedy_prune():
        # 1. Greedy Construction
        uncovered = np.ones(total_words, dtype=bool)
        selected = set()
        
        # Precompute list of all indices for shuffling tie-breaks
        all_indices = list(range(total_words))
        
        # Optimization: Instead of checking all 6561 every time, we can keep a list of "active" candidates?
        # Actually, 6561 is small enough that a numpy operation might be fast.
        # gain = M @ uncovered (dot product).
        # But M is sparse. 
        # Let's just loop if numpy dot is slow due to overhead? 
        # 6561 * 17 * 486 steps is the cost. 
        # If we use M @ uncovered, it's 6561*6561*486 operations in C. That's ~20 billion ops. Too slow.
        
        # So we must use the sparse structure.
        # gain[c] = sum(uncovered[x] for x in neighbors[c])
        
        # We can maintain gain array.
        gain = np.zeros(total_words, dtype=np.int32)
        # Initialize gain: since all are uncovered, gain[c] = len(neighbors[c]) = 17 for all c.
        gain[:] = 17
        
        # But wait, gain depends on `uncovered`.
        # When we add a codeword c, we set uncovered[neighbors[c]] = False.
        # This reduces gain for ALL codewords c' that share neighbors with c.
        # Updating gain for all c' is O(N * 17 * 17) per step? No.
        # For each x in neighbors[c] (the newly covered words), for each c' in "codewords covering x", gain[c'] -= 1.
        # We need `reverse_neighbors[x]` = list of codewords c' such that x in neighbors[c'].
        # This is the same as neighbors[x] because distance is symmetric.
        # So reverse_neighbors is just neighbors.
        
        # Algorithm:
        # 1. Init gain[c] = 17 for all c.
        # 2. Loop:
        #    a. Find c* with max gain. (If tie, pick randomly or first).
        #    b. If gain[c*] == 0, break (all covered).
        #    c. Add c* to selected.
        #    d. For each x in neighbors[c*]:
        #         If uncovered[x] is True:
        #             uncovered[x] = False
        #             For each c' in neighbors[x]:
        #                 gain[c'] -= 1
        
        # Complexity per step:
        # Finding max: O(N) = 6561.
        # Updating: 17 (neighbors of c*) * 17 (neighbors of x) = 289 operations.
        # Total steps: ~486.
        # Total ops: 486 * (6561 + 289) ~ 3.2M. Very fast.
        
        while True:
            if time.time() > deadline - 2.0:
                break
            # Find max gain
            # We can use numpy argmax on gain, but gain changes.
            # Since we need to handle ties or randomization, and gain is small int,
            # a simple python loop or numpy argmax is fine.
            # Numpy argmax is fast.
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Find all indices with max_gain? Or just one?
            # To diversify, we can pick randomly from the set of max_gain indices.
            # But creating a list of indices every time is O(N).
            # Let's just pick the first one found by argmax for now, or use a random tie-breaker.
            # For a "different strategy", deterministic greedy is the core. 
            # We will add a small randomization: if there are multiple max, pick one at random.
            
            # To avoid O(N) list creation, we can just use argmax.
            # If we want randomization, we can add a small random noise to gain?
            # No, let's stick to pure greedy first, then prune.
            
            c_star = np.argmax(gain)
            
            # Add to selected
            selected.add(c_star)
            
            # Update uncovered and gains
            # Get neighbors of c_star
            neigh_c = neighbors[c_star]
            
            for x in neigh_c:
                if uncovered[x]:
                    uncovered[x] = False
                    # Decrease gain for all codewords that cover x
                    # These are the neighbors of x
                    for c_prime in neighbors[x]:
                        gain[c_prime] -= 1
            
            # Optimization: We don't need to update gain for c_star itself? 
            # c_star covers x, so gain[c_star] decreases. But we already picked it.
            # It will be 0 eventually.
            
        # Now we have a valid code `selected`.
        # It might be larger than optimal.
        # Apply Pruning (Deletion)
        code_list = list(selected)
        is_code = np.zeros(total_words, dtype=bool)
        for idx in code_list:
            is_code[idx] = True
            
        cov_count = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            cov_count[neighbors[idx]] += 1
            
        # Prune
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle for random order
            random.shuffle(code_list)
            for c1 in code_list:
                # Check if removable
                removable = True
                for x in neighbors[c1]:
                    if cov_count[x] == 1:
                        removable = False
                        break
                if removable:
                    cov_count[neighbors[c1]] -= 1
                    is_code[c1] = False
                    code_list.remove(c1)
                    improved = True
        
        return code_list

    # Run a few restarts with slight variations?
    # The greedy algorithm is deterministic if we always pick argmax.
    # To get different results, we can introduce random tie-breaking.
    
    best_result = []
    
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
            
        # Reset state
        uncovered = np.ones(total_words, dtype=bool)
        selected = set()
        gain = np.full(total_words, 17, dtype=np.int32)
        
        while True:
            if time.time() > deadline - 5.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Random tie-breaking:
            # Find all indices with gain == max_gain
            # This is O(N).
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) > 1:
                c_star = candidates[random.randint(0, len(candidates) - 1)]
            else:
                c_star = candidates[0]
            
            selected.add(c_star)
            
            neigh_c = neighbors[c_star]
            for x in neigh_c:
                if uncovered[x]:
                    uncovered[x] = False
                    for c_prime in neighbors[x]:
                        gain[c_prime] -= 1
                        
        # Prune
        code_list = list(selected)
        is_code = np.zeros(total_words, dtype=bool)
        for idx in code_list:
            is_code[idx] = True
            
        cov_count = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            cov_count[neighbors[idx]] += 1
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            random.shuffle(code_list)
            for c1 in code_list:
                removable = True
                for x in neighbors[c1]:
                    if cov_count[x] == 1:
                        removable = False
                        break
                if removable:
                    cov_count[neighbors[c1]] -= 1
                    is_code[c1] = False
                    code_list.remove(c1)
                    improved = True
        
        # Verify and update best
        # Final check
        final_cov = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            final_cov[neighbors[idx]] += 1
            
        if np.all(final_cov > 0):
            if len(code_list) < len(best_result):
                best_result = code_list[:]
        
    # Fallback if no valid code found (should not happen)
    if not best_result:
        # Return a trivial valid code (all words)
        best_result = list(range(total_words))
        
    return [list(digits_of[i]) for i in best_result]
