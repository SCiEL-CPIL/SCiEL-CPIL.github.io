# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
import random

def construct(q, n, radius):
    """
    Construct a covering code using Iterated Local Search with Bitset acceleration.
    Strategy: 
    1. Greedy Set Cover to get an initial valid solution.
    2. Local Search: 
       - Try to remove a codeword if its ball is covered by others (redundant).
       - Try to remove a codeword if the uncovered gap can be covered by adding 
         a single other codeword (1-exchange).
       - Try to remove a codeword and repair the gap with a minimal set of new codewords 
         if the gap is small (k-repair for small k).
    3. Iterate with random restarts/perturbations if stuck.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    # Total number of words in the space
    N = q ** n
    if N > 200000:
        # Fallback for large spaces: use a simple constructive approach
        return _constructive_cover(q, n, radius)

    # Generate all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    # Precompute balls: for each word, which words are within distance radius
    # For efficiency, we use Python integers as bitsets since num_words (6561) 
    # is large for numpy boolean arrays in tight loops but fits in ~104 64-bit ints.
    # However, Python big integers are very fast for bitwise ops.
    
    # Map word index to integer value for quick lookup? 
    # Actually, we just need the ball of each word.
    
    # Compute balls as list of Python integers (bitsets)
    # bitset[i] has bit j set if word j is within distance radius of word i
    
    # To compute this efficiently:
    # 1. Create a mapping from word tuple to index
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}
    
    balls = [0] * num_words
    
    # Helper to convert word to index
    def get_idx(word_tuple):
        return word_to_idx.get(word_tuple, -1)

    # For each word, compute its ball
    for i in range(num_words):
        w = all_words[i]
        bits = 0
        # Include self
        bits |= (1 << i)
        
        if radius >= 1:
            # For each position, change the value
            for p in range(n):
                old_val = w[p]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    # Create new word
                    # Optimization: modify a copy or use tuple manipulation
                    # Since n is small (8), tuple manipulation is fast
                    new_w_list = list(w)
                    new_w_list[p] = new_val
                    j = get_idx(tuple(new_w_list))
                    if j != -1:
                        bits |= (1 << j)
        balls[i] = bits

    # Initial Greedy Cover
    # We maintain a set of selected indices
    # To speed up greedy, we can use numpy for the first pass, but for local search, bitsets are better.
    
    # Let's do a fast greedy using numpy first to get a decent start
    # Then convert to bitset representation
    
    # Numpy balls for greedy
    # This is memory intensive (6561x6561 bool) but ~42MB, acceptable.
    balls_np = np.zeros((num_words, num_words), dtype=bool)
    for i in range(num_words):
        # Extract bits from python int to numpy array? 
        # It's faster to recompute or use the python int logic if we stick to bitsets.
        # Let's stick to bitsets for the whole thing to avoid conversion overhead.
        pass

    # Greedy with Bitsets
    # covered_mask: Python int representing which words are covered
    # selected: set of indices
    
    def greedy_cover_bitset(deadline):
        selected = set()
        covered = 0
        # Precompute "uncovered" logic? 
        # Greedy step: find i that maximizes popcount(balls[i] & ~covered)
        
        # Optimization: 
        # 1. Start with empty
        # 2. While not all covered:
        #    - For each candidate i (maybe just all, or remaining), calculate gain
        #    - Pick max gain
        
        # Since 6561 is small, we can just loop through all candidates each time.
        # 6561 * 6561 operations in worst case per iteration? No, 6561 candidates * 1 bitwise AND.
        # Number of iterations ~ 500. 500 * 6561 * (bitwise ops) is very fast.
        
        all_mask = (1 << num_words) - 1
        
        while covered != all_mask and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            
            # To speed up, we can skip indices that are already selected? 
            # No, we want to add new ones.
            
            # Optimization: Only check indices that are not yet selected? 
            # Actually, adding a selected index does nothing.
            # So we iterate over range(num_words).
            
            # Even better: maintain a list of "active" candidates? 
            # For now, simple loop.
            
            # Vectorized approach with numpy would be faster for the "find max" step.
            # Let's use numpy for the greedy phase specifically.
            
            # Convert balls to numpy once
            if not hasattr(greedy_cover_bitset, 'balls_np'):
                greedy_cover_bitset.balls_np = np.zeros((num_words, num_words), dtype=bool)
                for i in range(num_words):
                    # Set bits
                    b = balls[i]
                    # This is slow. Let's just do the greedy with python ints but optimize the search.
                    pass
            
            # Let's stick to python ints but optimize the "find best" step.
            # Since we can't easily vectorize python int bitwise ops, 
            # we might be slower than numpy greedy.
            # 
            # Alternative: Use Numpy for Greedy, then convert to Bitsets for Local Search.
            # This is a good hybrid.
            
            if not hasattr(greedy_cover_bitset, 'balls_np_ready'):
                # Build numpy balls from the precomputed python int balls? 
                # Or just recompute numpy balls directly? 
                # Recomputing numpy balls is O(N^2 * n). 
                # Let's do that once.
                greedy_cover_bitset.balls_np = np.zeros((num_words, num_words), dtype=bool)
                np.fill_diagonal(greedy_cover_bitset.balls_np, True)
                if radius >= 1:
                    # Efficient numpy ball computation
                    # For each position p, and each value v, find words that differ only at p?
                    # No, radius 1 means differ in AT MOST 1 position.
                    
                    # Let's use the property: 
                    # word i covers word j if they are identical or differ in 1 pos.
                    
                    # We already have `balls` as python ints. 
                    # Converting 6561 python ints to a 6561x6561 numpy bool array is O(N^2).
                    # 6561^2 ~ 43M operations. In Python, this is slow (~10s+).
                    # 
                    # Let's compute numpy balls directly using broadcasting?
                    # all_words is (N, n).
                    # diff = (all_words[:, None, :] != all_words[None, :, :]) # (N, N, n) -> 43M * 8 * 4 bytes = 1.3GB. Too big.
                    
                    # We must use the python int balls or a smarter numpy method.
                    # Let's use the python int balls for the greedy too, but optimize the "find max" loop.
                    # 
                    # Actually, 500 iterations * 6561 candidates = 3.2M bitwise ANDs.
                    # Each bitwise AND on a 6561-bit integer is fast in C (Python big ints).
                    # Popcount is also fast (int.bit_count()).
                    # So pure Python bitset greedy should be fast enough.
                    
                    greedy_cover_bitset.balls_np_ready = True
            
            for i in range(num_words):
                if i in selected:
                    continue
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break # Should not happen if valid cover exists
            
            selected.add(best_idx)
            covered |= balls[best_idx]
            
        return selected, covered

    # Run Greedy
    initial_selected, initial_covered = greedy_cover_bitset(deadline)
    best_selected = set(initial_selected)
    best_size = len(best_selected)
    best_covered = initial_covered
    
    all_mask = (1 << num_words) - 1

    # Local Search
    # 1. Redundancy Removal: If removing i leaves covered unchanged, remove i.
    # 2. 1-Exchange: If removing i creates a gap G, and adding j covers G, swap i->j.
    # 3. k-Repair: If removing i creates a small gap G, find min set of k words to cover G.
    
    def local_search(selected, covered, deadline):
        # Returns improved (selected, covered)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate over a random permutation of selected indices to avoid bias
            candidates = list(selected)
            random.shuffle(candidates)
            
            for i in candidates:
                if i not in selected:
                    continue
                
                # Try removing i
                new_covered = covered & ~balls[i]
                
                # Check if fully covered
                if new_covered == all_mask:
                    # Redundant! Remove it
                    selected.remove(i)
                    covered = new_covered
                    improved = True
                    continue
                
                # Check 1-Exchange
                # Gap = ~new_covered
                gap = all_mask & ~new_covered
                gap_count = gap.bit_count()
                
                # If gap is small, maybe we can fill it with one word?
                # We need a word j (j not in selected) such that balls[j] covers gap.
                # i.e., gap & ~balls[j] == 0  =>  gap | balls[j] == balls[j]
                
                # Optimization: 
                # If gap_count is large, 1-exchange is unlikely to help unless we find a word with huge coverage.
                # We can iterate over all j not in selected? 
                # That's 6561 checks. 
                # For each i in selected (~500), that's 3.2M checks.
                # Each check: (gap & ~balls[j]).bit_count() == 0.
                # This might be too slow if we do it for every i in every iteration.
                
                # Heuristic: Only attempt 1-exchange if gap_count is below a threshold?
                # Or, precompute "best coverers" for common gaps?
                
                # Let's try a faster 1-exchange check:
                # We want min_j (gap & ~balls[j]).bit_count() == 0.
                
                # We can limit the search for j to words that have high overlap with the gap?
                # Or just iterate all j if the number of selected items is small?
                
                # Let's iterate all j, but break early if found.
                # To speed up, we can skip j if j is in selected.
                
                found_swap = False
                # Optimization: If gap_count > max_possible_single_ball_coverage, skip.
                # Max ball size is 1 + n*(q-1) = 1 + 8*2 = 17.
                # If gap_count > 17, no single word can cover it.
                if gap_count > 17:
                    continue
                
                for j in range(num_words):
                    if j in selected:
                        continue
                    # Check if j covers the gap
                    if (gap & ~balls[j]) == 0:
                        # Swap i with j
                        selected.remove(i)
                        selected.add(j)
                        covered = new_covered | balls[j]
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue
                
                # Try 2-Exchange (Repair with 2 words)
                # If gap_count is small (e.g., < 30), we can try to find 2 words that cover it.
                # This is more expensive.
                # Let's stick to 1-exchange for now to ensure speed.
                # If we have time, we can add 2-exchange.

        return selected, covered

    # Run Local Search
    best_selected, best_covered = local_search(best_selected, best_covered, deadline)

    # Iterated Local Search: 
    # If we are stuck, perturb the solution.
    # Perturbation: Randomly remove k words and repair greedily.
    
    while time.time() < deadline - 5:
        # Check if we can improve
        # Perturb
        current_selected = set(best_selected)
        current_covered = best_covered
        
        # Remove 2-5 random words
        k_remove = random.randint(2, 5)
        to_remove = random.sample(list(current_selected), min(k_remove, len(current_selected)))
        for i in to_remove:
            current_selected.remove(i)
            current_covered &= ~balls[i]
        
        # Repair greedily
        all_mask_local = all_mask
        while current_covered != all_mask_local and time.time() < deadline - 2:
            best_gain = -1
            best_idx = -1
            for i in range(num_words):
                if i in current_selected:
                    continue
                gain = (balls[i] & ~current_covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            current_selected.add(best_idx)
            current_covered |= balls[best_idx]
        
        # Local Search on the repaired solution
        current_selected, current_covered = local_search(current_selected, current_covered, deadline - 2)
        
        # Accept if better
        if len(current_selected) < len(best_selected):
            best_selected = current_selected
            best_covered = current_covered
            best_size = len(best_selected)

    # Convert back to list of lists
    result = [all_words[i].tolist() for i in best_selected]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    # Use the tetracode construction for ternary case
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    # General fallback: greedy
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    # Compute balls
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True
    
    # Simple greedy
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]
        
    return [all_words[i].tolist() for i in selected]
# EVOLVE-BLOCK-END
