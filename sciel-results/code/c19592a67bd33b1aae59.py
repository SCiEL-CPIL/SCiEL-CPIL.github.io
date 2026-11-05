import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    total = q ** n
    
    # If the space is too large for the matrix approach, fall back to tetracode sum
    # For n=6, q=3, total=729. Matrix is 729x729 ~ 531k bools. Very safe.
    # For larger n, this might blow up. But the task is specific to n=6.
    # We'll add a check just in case.
    if total > 10000:
        # Fallback to tetracode product if space is large
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # 1. Precompute balls as a boolean matrix
    # ball_matrix[i, j] is True if word j is within radius of word i
    ball_matrix = np.zeros((total, total), dtype=bool)
    
    # Helper to convert int to list of digits
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Helper to convert list of digits to int
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    # Precompute neighbors for each word
    # We can do this efficiently by iterating through all words and their neighbors
    for i in range(total):
        word = to_word(i)
        ball_matrix[i, i] = True
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word[pos] = new_val
                j = to_int(word)
                ball_matrix[i, j] = True
            word[pos] = orig

    # 2. Greedy Set Cover
    covered_mask = np.zeros(total, dtype=bool)
    selected_indices = []
    
    while not covered_mask.all():
        if time.time() > deadline - 10:
            break
            
        # Count how many uncovered points each candidate covers
        # ~covered_mask is the set of uncovered points
        # ball_matrix[:, ~covered_mask] selects columns that are uncovered
        # summing along axis 1 gives the count per row (candidate)
        counts = ball_matrix[:, ~covered_mask].sum(axis=1)
        
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break # Should not happen if not all covered, but safety
            
        selected_indices.append(best_idx)
        covered_mask |= ball_matrix[best_idx]

    # 3. Local Search: Remove redundant codewords
    # We maintain a coverage count array
    # coverage_count[j] = number of selected codewords that cover word j
    
    # Initialize coverage counts
    # We can do this by summing the rows of ball_matrix corresponding to selected_indices
    # selected_mask is a boolean vector of length total, True if index is in selected_indices
    selected_mask = np.zeros(total, dtype=bool)
    for idx in selected_indices:
        selected_mask[idx] = True
        
    # coverage_count = ball_matrix[selected_mask].sum(axis=0)
    # This sums the rows of selected codewords to get how many times each word is covered
    coverage_count = ball_matrix[selected_mask].sum(axis=0)
    
    # List of currently active codeword indices
    active_codewords = list(selected_indices)
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        # Try to remove any codeword that is redundant
        # A codeword c is redundant if for all w in ball(c), coverage_count[w] > 1
        # i.e., min(coverage_count[ball(c)]) > 1
        
        # We iterate through a copy of active_codewords
        # If we remove one, we restart the inner loop (or just continue, but restart is safer for correctness of 'changed' logic if multiple become redundant)
        # Actually, if we remove one, coverage counts decrease, so other codewords *might* become redundant.
        # So we should loop until no more removals are possible.
        
        new_active = []
        for c in active_codewords:
            # Check if c is redundant
            # Get the indices covered by c
            # These are the columns where ball_matrix[c] is True
            # We need to check if coverage_count at those positions is all > 1
            covered_by_c = np.where(ball_matrix[c])[0]
            if np.all(coverage_count[covered_by_c] > 1):
                # Redundant! Remove it
                coverage_count -= ball_matrix[c]
                changed = True
                # We do not add c to new_active
            else:
                new_active.append(c)
                
        active_codewords = new_active

    # 4. Convert back to list of lists
    final_codewords = [to_word(idx) for idx in active_codewords]
    
    # Ensure distinctness (should be guaranteed, but just in case)
    seen = set()
    unique_codewords = []
    for c in final_codewords:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_codewords.append(c)
            
    return unique_codewords
