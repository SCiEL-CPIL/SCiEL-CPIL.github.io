import itertools
import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code for q=3, n=6, radius=1.
    Uses the direct sum of the tetracode as a candidate pool and applies
    a greedy set cover algorithm to minimize the number of codewords.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (though task specifies this case)
        # Simple exhaustive or random construction
        return _fallback_construct(q, n, radius)

    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Generate the 81 candidate codewords from the tetracode direct sum
    # Tetracode: 9 words of length 4 over {0,1,2}
    # t = [a, b, (a+b)%3, (a+2b)%3]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    candidates = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            candidates.append(t + list(tail))
            
    # 2. Precompute neighbors (Hamming distance <= 1) for each candidate
    # A word w covers itself and words differing in 1 position
    # For n=6, q=3, number of neighbors = 1 + 6*2 = 13
    candidate_indices = list(range(len(candidates)))
    candidate_sets = []
    
    # Map word to index for fast lookup?
    # Total words = 3^6 = 729.
    # We can represent words as integers 0..728 for fast set operations.
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute all 729 words as indices
    all_words = set(range(729))
    
    # Precompute neighbors for each candidate
    # neighbors[i] is a set of indices covered by candidates[i]
    neighbors = []
    for c in candidates:
        covered = set()
        covered.add(word_to_idx(c))
        for pos in range(n):
            for val in range(q):
                if val == c[pos]:
                    continue
                new_c = c[:]
                new_c[pos] = val
                covered.add(word_to_idx(new_c))
        neighbors.append(covered)

    # 3. Greedy Set Cover
    # We want to cover all_words using the fewest neighbors from the candidate pool.
    uncovered = all_words.copy()
    selected = []
    available = set(candidate_indices)
    
    # Optimization: Precompute for each word, which candidates cover it?
    # word_to_candidates = {word_idx: set of candidate indices}
    word_to_candidates = {w: set() for w in all_words}
    for i, c_set in enumerate(neighbors):
        for w in c_set:
            word_to_candidates[w].add(i)
            
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find candidate that covers most uncovered words
        best_candidate = None
        best_count = -1
        
        # Iterate through available candidates
        # To speed up, we can iterate through uncovered words and count frequencies of their covering candidates
        # But 81 candidates is small.
        for i in available:
            if i not in neighbors: continue # Should not happen
            count = 0
            # Count how many words in neighbors[i] are in uncovered
            # Using intersection
            # len(neighbors[i] & uncovered)
            # This is fast in Python sets
            count = len(neighbors[i] & uncovered)
            if count > best_count:
                best_count = count
                best_candidate = i
                
        if best_candidate is None:
            # This shouldn't happen if the pool covers everything
            break
            
        selected.append(best_candidate)
        available.remove(best_candidate)
        uncovered -= neighbors[best_candidate]
        
    # If we ran out of time or broke early, check if we have a valid cover
    # If not, we need to ensure validity.
    # The greedy algorithm guarantees validity if it completes.
    # If it breaks due to time, `uncovered` might not be empty.
    # We must return a valid code.
    # If time ran out, we can just return the full 81 candidate set? 
    # No, that's 81. We want fewer.
    # But 81 is valid.
    # However, the loop is very fast (81 * 729 operations approx). It should finish in milliseconds.
    
    # 4. Local Search to potentially improve (reduce size)
    # Try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other selected codewords.
    
    # Build coverage map for selected codewords
    # word_to_cover_count = {w: count of selected codewords covering w}
    
    def is_valid_cover(selected_indices):
        if not selected_indices:
            return len(all_words) == 0
        covered = set()
        for i in selected_indices:
            covered |= neighbors[i]
        return covered == all_words

    # Greedy result is valid.
    # Try to remove redundant ones.
    changed = True
    while changed:
        changed = False
        # Calculate coverage counts
        cover_count = {w: 0 for w in all_words}
        for i in selected:
            for w in neighbors[i]:
                cover_count[w] += 1
        
        # Try to remove each selected codeword
        for i in list(selected):
            if time.time() > deadline:
                break
            # Check if removing i leaves any word uncovered
            # i.e., if there is any w in neighbors[i] with cover_count[w] == 1
            redundant = True
            for w in neighbors[i]:
                if cover_count[w] == 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                changed = True
                # Recalculate cover_count for next iteration? 
                # Yes, or break inner loop and restart outer while.
                break
                
    # Final check
    if not is_valid_cover(selected):
        # Fallback to full set if something went wrong
        selected = candidate_indices

    # Convert indices back to codewords
    result = [candidates[i] for i in selected]
    
    # Ensure distinctness (should be distinct by construction)
    # And validity
    if not is_valid_cover(selected):
         # Emergency fallback: return all 81
         result = candidates
         
    return result

def _fallback_construct(q, n, radius):
    # Simple random greedy for other cases
    import random
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    uncovered = set(all_words)
    selected = []
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
        
    while uncovered:
        # Pick random word from uncovered? Or best candidate?
        # For simplicity, pick a random word from uncovered as a new codeword
        # This is a valid covering code construction (greedy by picking uncovered words)
        w = random.choice(list(uncovered))
        selected.append(w)
        # Remove all words within radius
        to_remove = []
        for u in uncovered:
            if hamming_dist(w, u) <= radius:
                to_remove.append(u)
        for u in to_remove:
            uncovered.remove(u)
            
    return selected
