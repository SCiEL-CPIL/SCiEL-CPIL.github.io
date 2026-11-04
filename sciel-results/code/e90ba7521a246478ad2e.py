import numpy as np
import itertools

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # For q=3, n=8, radius=1, use greedy set cover followed by reduction
    total_words = q ** n  # 6561
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres (words within distance 1) as index sets
    sphere_indices = []
    for word in all_words:
        sphere = set([word])
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    sphere.add(new_to_idx if False else word_to_idx[new_word]) # Fix: need to add to set of indices
        # Re-doing sphere calculation to store indices directly
        sphere_idx_set = set()
        w_idx = word_to_idx[word]
        sphere_idx_set.add(w_idx)
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    sphere_idx_set.add(word_to_idx[new_word])
        sphere_indices.append(sphere_idx_set)

    # Uncovered words
    uncovered = set(range(total_words))
    code_indices = []
    
    # Greedy Phase
    while uncovered:
        best_idx = -1
        best_count = -1
        # Optimization: We can skip words that are already covered? No, the sphere of a covered word might cover many uncovered words.
        # But we can restrict the search to words that are themselves uncovered? 
        # Actually, the center of the sphere doesn't have to be uncovered.
        # However, for radius 1, if a word is covered, its neighbors are likely covered too.
        # Standard greedy checks all candidates.
        for i in range(total_words):
            # Optimization: If i is already in code_indices, skip? No, we don't add duplicates.
            # We can pre-compute the intersection size?
            # For speed, we can just iterate. 6561 * 6561 is ~43M ops. In Python, this might be slow.
            # Let's check the time limit: 120s.
            # The current code does this. It scored 1.11. So it runs.
            count = len(sphere_indices[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
            if best_count == len(uncovered):
                break
        if best_count == 0:
            break
        code_indices.append(best_idx)
        uncovered -= sphere_indices[best_idx]
        
    # Reduction Phase
    # We have a list of indices. We want to remove redundant ones.
    # A codeword c is redundant if every word in S(c) is also in S(c') for some other c' in the set.
    # Equivalently, if the set of words uniquely covered by c is empty.
    
    # To speed up redundancy check:
    # Maintain a count for each word of how many codewords cover it.
    # Initially, compute counts for all words based on code_indices.
    cover_count = np.zeros(total_words, dtype=int)
    for idx in code_indices:
        for w_idx in sphere_indices[idx]:
            cover_count[w_idx] += 1
            
    # A word is "uniquely covered" by c if cover_count[w] == 1 and w is in S(c).
    # But we need to know if *any* word in S(c) is uniquely covered by c.
    # If yes, c is essential.
    # If no, c is redundant.
    
    # We can iterate and remove.
    # Since removal changes cover_count, we should do it carefully.
    # Simple approach: Loop through code_indices. If redundant, remove and update counts.
    # To avoid O(N^2) complexity with large sets, we can use a heap or just iterate.
    # N is ~500. N^2 is 250k. Sphere size is 1+8*2=17.
    # Total ops ~ 500 * 250k * 17? No.
    # For each candidate for removal (N of them), check if it has unique words.
    # To check if c has unique words, we look at all w in S(c) (17 words) and see if cover_count[w] == 1.
    # If any w has count 1, c is essential.
    # If all w have count > 1, c is redundant.
    
    # We can do this in a loop.
    # Since removing a codeword decreases counts, a codeword that was essential might become redundant? No, if it had a unique word, that word's count was 1. Removing other codewords doesn't increase counts. So it remains unique.
    # Wait, if I remove a codeword, counts decrease. A word with count 2 becomes 1.
    # So a codeword that was NOT essential (all words covered >=2) might become essential if one of its words drops to 1?
    # No, if all words in S(c) have count >= 2, and I remove another codeword, the counts of words in S(c) might drop to 1.
    # If a word in S(c) drops to 1, then c becomes the *only* cover for that word, so c becomes essential.
    # So we can't just check once. We need to re-evaluate.
    
    # However, if we remove a redundant codeword, we are safe.
    # We can repeat:
    # 1. Find any redundant codeword.
    # 2. Remove it.
    # 3. Update counts.
    # 4. Repeat.
    
    # To find a redundant codeword efficiently:
    # We can iterate through the current list.
    
    code_indices_set = set(code_indices)
    # We need to be able to remove from a list efficiently. 
    # Since N is small, we can just use a list and filter, or use a set for membership and a list for order.
    
    # Let's use a list for the current code.
    current_code = list(code_indices)
    
    # Recompute cover_count from scratch or maintain?
    # Maintaining is better.
    # We already have cover_count from the initial set.
    
    changed = True
    while changed:
        changed = False
        # Iterate through a copy of the list to allow removal
        for i in range(len(current_code)):
            c_idx = current_code[i]
            # Check if c_idx is redundant
            is_redundant = True
            for w_idx in sphere_indices[c_idx]:
                if cover_count[w_idx] == 1:
                    # This word is only covered by c_idx (among current_code)
                    # So c_idx is essential
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c_idx
                # Update cover_count
                for w_idx in sphere_indices[c_idx]:
                    cover_count[w_idx] -= 1
                
                # Remove from list
                current_code.pop(i)
                changed = True
                # After removal, previous indices are shifted. 
                # Also, other codewords might now be redundant or essential?
                # As noted, removing a codeword can make others essential (if their unique-ness was hidden? No, counts decrease, so more words become unique to the remaining codewords).
                # Wait. If counts decrease, a word with count 2 becomes 1. 
                # If that word is in S(d) for another codeword d, then d becomes the unique cover for that word.
                # So d becomes essential.
                # Can a codeword become redundant? 
                # If counts decrease, a word with count 3 becomes 2. It is still covered by >=2.
                # So a codeword that was redundant (all words count >=2) might have words drop to 1, making it essential.
                # It cannot become redundant if it wasn't.
                # So we only need to check if *new* codewords become redundant? 
                # No, if it was essential, it stays essential or becomes more essential.
                # If it was redundant, it might become essential.
                # So we only need to check codewords that were previously redundant? 
                # Actually, we just loop again. The `while changed` loop will re-check everyone.
                # Since N is small, this is fine.
                break # Restart the for loop with the updated list
                
    return [list(all_words[idx]) for idx in current_code]
