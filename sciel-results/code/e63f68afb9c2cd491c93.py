import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)

    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
    
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    
    # Maintain coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1
            
    # Helper function to check if a codeword is redundant
    def is_redundant(j, cov):
        for nb in neighbors[j]:
            if cov[nb] == 1:
                return False
        return True

    # Helper function to try to reduce a set
    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    for nb in neighbors[j]:
                        cov[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set

    # Local Search with Simulated Annealing / Random Walk
    best_selected = list(selected)
    best_cov = coverage_count.copy()
    
    # First, try to reduce the initial greedy solution
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    random.seed(42)
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We track the "energy" as the negative of the sum of coverage counts?
    # Or just use the sum of coverage counts as a measure of "slack".
    # Higher slack is generally better for finding removals.
    # Let's define energy = -sum(coverage_count). We want to minimize energy (maximize slack).
    # Or simpler: Accept same-size swaps with a fixed probability to randomize.
    # Let's try: Accept if valid. If size decreases, great. If size same, accept with prob 0.5.
    # This performs a random walk on the space of valid covers of size k.
    
    current_size = len(best_selected)
    
    # To speed up, we precompute the list of neighbors for fast access? 
    # neighbors is already a list of lists.
    
    # Optimization: Use a set for selected for O(1) lookup? 
    # selected is a list. We can maintain a set in parallel.
    selected_set = set(best_selected)
    
    while time.time() < deadline - 2.0:
        # 1. Try to remove
        new_selected = try_reduce(best_selected, best_cov, deadline)
        if len(new_selected) < current_size:
            current_size = len(new_selected)
            best_selected = new_selected
            selected_set = set(best_selected)
            # Restart T? Or just continue.
            T = 1.0 # Reset temperature to explore again
            continue
            
        # 2. Try a swap
        if len(best_selected) == 0:
            break
            
        # Pick random i from selected
        i_idx = random.randint(0, len(best_selected) - 1)
        i_word = best_selected[i_idx]
        
        # Pick random j not in selected
        # To avoid O(N) scan for "not in set", we can just pick random j and check.
        # If N=729, set size ~73, prob of collision is ~10%. Acceptable.
        j_word = random.randint(0, N - 1)
        if j_word in selected_set:
            continue
            
        # Attempt swap i -> j
        # 1. Remove i
        for nb in neighbors[i_word]:
            best_cov[nb] -= 1
            
        # Check validity
        valid = True
        for nb in neighbors[i_word]:
            if best_cov[nb] < 0: # Should be 0 if it was 1 and we removed the only cover
                valid = False
                break
            if best_cov[nb] == 0:
                # It's covered by 0 now? No, if it was 1, it becomes 0. 
                # If it was 1, it becomes 0. Is 0 allowed? 
                # The definition of covering code: every word must be within distance radius.
                # If coverage count is 0, it is NOT covered.
                # So if any neighbor of i had count 1, it becomes 0, which is invalid.
                valid = False
                break
                
        if not valid:
            # Revert
            for nb in neighbors[i_word]:
                best_cov[nb] += 1
            continue
            
        # 2. Add j
        for nb in neighbors[j_word]:
            best_cov[nb] += 1
            
        # Now the set is valid (size is same)
        # Try to reduce
        # We need to update best_selected list for try_reduce
        # try_reduce modifies the list in place. 
        # We should make a copy or handle carefully.
        # Let's make a copy of the list, remove i, add j.
        temp_selected = list(best_selected)
        temp_selected[i_idx] = j_word
        
        # Note: try_reduce expects the list to match the coverage counts.
        # Our best_cov is currently updated for the swap.
        # So we can pass temp_selected and best_cov.
        
        new_selected = try_reduce(temp_selected, best_cov, deadline)
        
        # Check if size decreased
        if len(new_selected) < current_size:
            # Accept
            best_selected = new_selected
            selected_set = set(best_selected)
            current_size = len(new_selected)
            T = 1.0
            # Note: best_cov is already updated for the swap and reductions.
            # But wait, try_reduce modified best_cov.
            # We need to ensure best_cov corresponds to best_selected.
            # try_reduce does that.
            continue
        else:
            # Size is same (or increased? No, try_reduce only removes).
            # So size is same.
            # Accept this state with some probability to randomize the walk.
            # If we accept, we update best_selected to temp_selected (reduced version).
            # And best_cov is already correct.
            
            # Simple random walk: accept with prob 0.5
            if random.random() < 0.5:
                best_selected = new_selected
                selected_set = set(best_selected)
            else:
                # Revert the swap in best_cov
                # Remove j
                for nb in neighbors[j_word]:
                    best_cov[nb] -= 1
                # Add i
                for nb in neighbors[i_word]:
                    best_cov[nb] += 1
                # best_selected remains unchanged
                
        T *= cooling
        if T < min_T:
            T = min_T

    # Final check
    codewords = [words[i] for i in best_selected]
    return codewords
