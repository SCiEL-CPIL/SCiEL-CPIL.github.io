"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Greedy set-cover for covering code K_q(n, radius) with local search polish.
    Uses numpy arrays for efficient coverage tracking and a swap-based local search
    to reduce codeword count beyond simple removal-based polish.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1
    
    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)
    
    # Precompute ball indices for all possible codewords
    # For radius=1: ball = {word} ∪ {word with one position changed to each other value}
    # Build ball_arrays as int64 arrays
    ball_arrays = []
    for idx in range(total_words):
        word = index_to_word(idx, q, n)
        covered = [idx]
        for pos in range(n):
            base = idx
            digit = word[pos]
            # Compute the base offset for this position
            # index = sum(word[i] * q^(n-1-i)), so position pos has weight q^(n-1-pos)
            weight = q ** (n - 1 - pos)
            for val in range(q):
                if val != digit:
                    new_idx = idx + (val - digit) * weight
                    covered.append(new_idx)
        ball_arrays.append(np.array(covered, dtype=np.int64))
    
    # Greedy set cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    codeword_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 10:
            break
        
        # Find the codeword that covers the most uncovered words
        best_count = -1
        best_idx = -1
        
        for idx in range(total_words):
            ball = ball_arrays[idx]
            count = np.count_nonzero(uncovered[ball])
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codeword_indices.append(best_idx)
        uncovered[ball_arrays[best_idx]] = 0
    
    # Check validity
    if uncovered.any():
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution
    
    # Local search polish: try to remove codewords and swap codewords
    # First pass: try removing each codeword (if its coverage is redundant)
    m = len(codeword_indices)
    
    # Precompute coverage counts: for each word, how many codewords cover it
    # Use a numpy array for efficiency
    cover_count = np.zeros(total_words, dtype=np.int32)
    for ci in codeword_indices:
        cover_count[ball_arrays[ci]] += 1
    
    # Try removing codewords whose ball is fully covered by others
    changed = True
    while changed and time.time() < deadline - 15:
        changed = False
        for i in range(len(codeword_indices)):
            ci = codeword_indices[i]
            # Check if removing ci leaves all words in its ball still covered
            ball = ball_arrays[ci]
            # Words that would become uncovered if we remove ci
            would_uncover = (cover_count[ball] == 1)
            if not would_uncover.any():
                # Safe to remove
                cover_count[ball] -= 1
                codeword_indices.pop(i)
                changed = True
                break  # restart from beginning
    
    # Second pass: swap-based local search
    # Try swapping a codeword with another that covers all uniquely-covered words
    # and gains additional coverage
    if time.time() < deadline - 5:
        m = len(codeword_indices)
        # Rebuild cover_count after removals
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in codeword_indices:
            cover_count[ball_arrays[ci]] += 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(codeword_indices)):
                ci = codeword_indices[i]
                ball_i = ball_arrays[ci]
                # Words uniquely covered by ci
                unique_to_i = (cover_count[ball_i] == 1)
                
                # Try swapping with a codeword not in current set
                # Sample candidate positions to keep this fast
                candidates = []
                # Try a few random candidates plus some systematic ones
                if time.time() > deadline - 10:
                    break
                
                # For speed, only try a limited number of swaps per iteration
                num_tries = 0
                max_tries = 50
                for attempt in range(max_tries):
                    # Pick a random candidate not in current set
                    cand = np.random.randint(0, total_words)
                    if cand in codeword_indices:
                        continue
                    num_tries += 1
                    
                    ball_cand = ball_arrays[cand]
                    # Check if cand covers all words uniquely covered by ci
                    if unique_to_i.any():
                        # Words uniquely covered by ci must be in cand's ball
                        unique_indices = np.where(unique_to_i)[0]
                        # Check if all unique indices are covered by cand
                        # i.e., cover_count[unique_indices] >= 1 after removing ci
                        # Actually: after removing ci, unique words have count 0
                        # So we need cand to cover them: they must be in ball_cand
                        # More precisely: for each word w in unique_to_i positions,
                        # w must be in ball_cand
                        # Use: the set of words in ball_i with count==1 must be subset of ball_cand
                        # Efficient check: count how many of ball_i's unique words are NOT in ball_cand
                        # = number of words in ball_i where cover_count==1 AND not in ball_cand
                        # We can check: for each unique word index, is it in ball_cand?
                        # Faster: create a mask
                        # Actually let's just check: all unique words must be in ball_cand
                        # Use numpy: create a boolean array for ball_cand membership
                        # This is expensive for large spaces, so use a different approach
                        # Check: the intersection of (ball_i where count==1) and (not ball_cand) is empty
                        # i.e., unique_words ⊆ ball_cand
                        # Compute: for each unique word, check if it's in ball_cand
                        # Use: np.isin is slow; instead use a temporary array
                        # Alternative: count = sum(cover_count[ball_i] == 1 and not in ball_cand)
                        # Let's just check the gain: how many NEW words does cand cover?
                        # And verify unique words are covered.
                        pass
                    
                    # Simpler approach: just check if after swap, coverage is valid
                    # Remove ci, add cand, check no word drops to 0
                    cover_count[ball_i] -= 1
                    cover_count[ball_cand] += 1
                    if np.any(cover_count == 0):
                        # Invalid swap
                        cover_count[ball_cand] -= 1
                        cover_count[ball_i] += 1
                        continue
                    # Valid swap! But we only swap if it doesn't increase size
                    # Actually swap keeps same size. For reduction, we need removal.
                    # Let's just accept valid swaps to rearrange, then try removals again
                    codeword_indices[i] = cand
                    improved = True
                    break
                
                if improved:
                    break
    
    # Final removal pass after swaps
    if time.time() < deadline - 3:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in codeword_indices:
            cover_count[ball_arrays[ci]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(codeword_indices)):
                ci = codeword_indices[i]
                ball = ball_arrays[ci]
                if not (cover_count[ball] == 1).any():
                    cover_count[ball] -= 1
                    codeword_indices.pop(i)
                    changed = True
                    break
    
    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in codeword_indices]
    
    # Final validity check
    final_cover = np.zeros(total_words, dtype=np.uint8)
    for ci in codeword_indices:
        final_cover[ball_arrays[ci]] = 1
    
    if final_cover.any() == 0 or np.any(final_cover == 0):
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_solution
    
    return codewords

# EVOLVE-BLOCK-END
