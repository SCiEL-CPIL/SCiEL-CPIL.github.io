"""Covering code K_3(8,1) via greedy set cover + local search."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    total = q ** n
    # Each codeword c covers c and all words at Hamming distance <= radius
    # For radius=1, q=3, n=8: covers 1 + n*(q-1) = 17 words

    # Precompute: for each codeword index, which words it covers
    # Use numpy for speed. Represent words as integers 0..total-1.
    # A word at index w has digits via base-q expansion.

    # For greedy: pick codeword covering most uncovered words.
    # Key insight: two words are at distance 1 if they differ in exactly one position.
    # For a given word w, the set of words at distance 1 from w are:
    #   for each position i, for each value v != w[i]: word with v at position i.

    # Strategy: 
    # 1. Greedy set cover to get initial solution
    # 2. Local search: try removing codewords if still covered, try swapping

    # Represent all words as a 2D array of shape (total, n)
    # Actually, for memory: total = 6561, n = 8, that's fine.
    words = np.array(list(np.array(np.unravel_index(np.arange(total), (q,)*n)).T), dtype=np.int32)
    # words[i] is the i-th word as a tuple of n digits

    # For greedy: maintain uncovered set. Initially all uncovered.
    # For each candidate codeword c, compute how many uncovered words it covers.
    # The words covered by c are: c itself + all words at distance 1 from c.

    # Precompute for each word index, the list of word indices at distance 1.
    # This is a (total, 17) array for radius=1, q=3, n=8.
    
    # Actually, let's think differently. For greedy set cover on 6561 elements,
    # we need to evaluate 6561 candidate sets per iteration, each of size 17.
    # That's 6561*17 = ~111k operations per iteration. With ~500 iterations, that's ~55M ops.
    # In numpy this should be fast.

    # Precompute: for each codeword index c, the set of indices it covers.
    # cover[c] = set of word indices at distance <= radius from c.
    
    # For radius=1: cover[c] = {c} union {words at distance 1 from c}
    # Words at distance 1 from c: for each position i, for each v in 0..q-1, v != c[i]:
    #   word with c[i] replaced by v.
    
    # Let's precompute the cover sets as a boolean matrix or list of sets.
    # total = 6561, each cover has 17 elements.
    
    # Better: represent coverage as a bitset or use numpy.
    # Let's use a 2D boolean array: cover_matrix[c, w] = True if w is covered by c.
    # That's 6561 x 6561 = 43M booleans = 43MB. Acceptable.
    
    cover_matrix = np.zeros((total, total), dtype=bool)
    
    # For each codeword c, mark c and its distance-1 neighbors.
    for c_idx in range(total):
        cover_matrix[c_idx, c_idx] = True
        c_word = words[c_idx]
        for i in range(n):
            for v in range(q):
                if v == c_word[i]:
                    continue
                # Create the word with position i changed to v
                new_word = c_word.copy()
                new_word[i] = v
                # Find the index of this word
                # We need a mapping from word tuple to index.
                # Precompute: index = sum(word[i] * q^(n-1-i))
                w_idx = 0
                for j in range(n):
                    w_idx = w_idx * q + new_word[j]
                cover_matrix[c_idx, w_idx] = True
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # For each candidate c, count how many uncovered words it covers
        # coverage_count[c] = sum(cover_matrix[c, :][uncovered])
        coverage_count = cover_matrix[:, uncovered].sum(axis=1)
        
        # Pick the candidate with maximum coverage
        best_c = np.argmax(coverage_count)
        if coverage_count[best_c] == 0:
            break
        
        selected.append(best_c)
        uncovered[cover_matrix[best_c, :]] = False
    
    # Convert selected indices to codeword lists
    codewords = [words[c].tolist() for c in selected]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return codewords
    
    # Local search: try to remove codewords
    # For each selected codeword, check if removing it leaves all words still covered.
    # Build a coverage count: for each word, how many selected codewords cover it.
    
    def compute_coverage_counts(sel_list):
        counts = np.zeros(total, dtype=np.int32)
        for c in sel_list:
            counts[cover_matrix[c, :]] += 1
        return counts
    
    counts = compute_coverage_counts(selected)
    
    # Try removing each codeword
    changed = True
    while changed and (time.time() - start_time) < deadline - 10:
        changed = False
        for idx in range(len(selected)):
            if (time.time() - start_time) > deadline - 10:
                break
            c = selected[idx]
            # Check if removing c leaves any word uncovered
            # A word becomes uncovered if counts[word] == 1 and c covers word
            # i.e., if any word has counts[word] == 1 and cover_matrix[c, word] == True
            # More precisely: after removing c, words that were only covered by c become uncovered.
            # So check: is there any word w such that cover_matrix[c, w] and counts[w] == 1?
            if not (cover_matrix[c, :] & (counts == 1)).any():
                # Safe to remove
                selected.pop(idx)
                counts[cover_matrix[c, :]] -= 1
                changed = True
                break  # Restart the scan since indices changed
    
    # Try swapping: replace one codeword with another that covers more uncovered... 
    # Actually after removal, everything is covered. Let's try to reduce further.
    
    # Try adding a codeword that allows removing multiple others? Complex.
    # Instead, try: for each pair (remove c1, add c2), check if it reduces size.
    # That's O(m^2 * 17) which for m~500 is 250k*17 = 4.25M per pass. Might be slow.
    
    # Simpler: try to replace one codeword c with another c' such that c' covers all words
    # that only c covers, and c' is not already selected.
    
    if (time.time() - start_time) < deadline - 5:
        counts = compute_coverage_counts(selected)
        sel_set = set(selected)
        for idx in range(len(selected)):
            if (time.time() - start_time) > deadline - 5:
                break
            c = selected[idx]
            # Words that only c covers
            only_c = cover_matrix[c, :] & (counts == 1)
            if not only_c.any():
                continue
            # Try to find c' not in sel_set such that c' covers all words in only_c
            # For each candidate c', check if cover_matrix[c', only_c_words].all()
            only_c_indices = np.where(only_c)[0]
            # Check candidates
            for c2 in range(total):
                if c2 in sel_set:
                    continue
                if cover_matrix[c2, only_c_indices].all():
                    # Replace c with c2
                    selected[idx] = c2
                    sel_set.discard(c)
                    sel_set.add(c2)
                    counts[cover_matrix[c, :]] -= 1
                    counts[cover_matrix[c2, :]] += 1
                    changed = True
                    break
    
    codewords = [words[c].tolist() for c in selected]
    return codewords
