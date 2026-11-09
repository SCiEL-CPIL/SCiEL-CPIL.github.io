"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import deque

def construct(q, n, radius):
    # For the specific task: q=3, n=8, radius=1
    # We will implement a greedy set cover with a removal phase.
    # Space size is 3^8 = 6561.
    # Each codeword covers 1 + 8*(3-1) = 17 words (radius 1 ball).
    # Wait, for q=3, radius 1: 1 + n*(q-1) = 1 + 8*2 = 17.
    # The seed uses 486.
    # We can represent words as integers 0..6560.
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 120s limit
    
    total_words = q ** n
    
    # Precompute all words as integers
    # We need a way to convert int <-> tuple
    # For q=3, n=8, this is efficient enough.
    
    # Helper: int to word tuple
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[n - 1 - i] = x % q
            x //= q
        return tuple(w)
    
    # Helper: word tuple to int
    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x
    
    # Precompute balls for each word (the set of words covered by that codeword)
    # ball[c] = list of word indices covered by c
    # We also need the "neighbors" or "incidence" structure for fast updates.
    # For each word w, list of codewords c such that w is in ball[c].
    # This is symmetric: w is in ball[c] iff c is in ball[w] (Hamming distance is symmetric).
    
    balls = [[] for _ in range(total_words)]
    
    # Generate all words
    all_words = [int_to_word(i) for i in range(total_words)]
    
    # Precompute balls
    for c_idx in range(total_words):
        c_word = all_words[c_idx]
        # Cover c_word itself
        balls[c_idx].append(c_idx)
        # Cover neighbors (Hamming distance 1)
        for pos in range(n):
            orig = c_word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(c_word)
                new_word[pos] = val
                neighbor_idx = word_to_int(tuple(new_word))
                balls[c_idx].append(neighbor_idx)
    
    # Incidence: for each word w, which codewords cover it?
    # Since distance is symmetric, cover_inc[w] = [c for c in range(total_words) if w in balls[c]]
    # This is the same as balls[w] if we interpret balls[c] as "c covers these words".
    # Wait. balls[c] is the set of words covered by c.
    # Let `covered_by[w]` be the list of codewords that cover w.
    # If w is in balls[c], then c is in covered_by[w].
    # Due to symmetry of Hamming distance, c is in balls[w] iff w is in balls[c].
    # So `covered_by[w]` is exactly `balls[w]`.
    
    # Greedy Set Cover
    # State:
    # uncovered: boolean array of size total_words, initially all True.
    # cov_count: for each codeword c, number of currently uncovered words in balls[c].
    
    uncovered = [True] * total_words
    cov_count = [len(balls[c]) for c in range(total_words)]
    
    selected = []
    
    # To speed up finding max, we could use a heap, but total_words is only 6561.
    # Scanning 6561 integers is ~1-2 ms.
    # Number of steps is likely ~300-500.
    # Total time ~ 500 * 2ms = 1s. Very fast.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Find codeword with max cov_count
        best_idx = -1
        max_count = -1
        for c in range(total_words):
            if cov_count[c] > max_count:
                max_count = cov_count[c]
                best_idx = c
        
        if max_count <= 0:
            break
        
        selected.append(best_idx)
        
        # Update uncovered and cov_count
        for w in balls[best_idx]:
            if uncovered[w]:
                uncovered[w] = False
                # For each codeword c' that covers w, decrement cov_count[c']
                for c_prime in balls[w]:
                    cov_count[c_prime] -= 1
                    
    # Removal Phase
    # We have a valid cover `selected`.
    # We try to remove codewords one by one.
    # Maintain `word_cover_count[w]`: how many selected codewords cover w.
    
    word_cover_count = [0] * total_words
    for c in selected:
        for w in balls[c]:
            word_cover_count[w] += 1
            
    # To allow removal, we iterate through selected (maybe in random order or reverse order of selection)
    # If we remove c, for each w in balls[c], word_cover_count[w] decreases by 1.
    # If any becomes 0, c cannot be removed.
    
    # Strategy: Iterate through selected. Try to remove. If possible, remove and update.
    # Since removal changes the counts, we might want to repeat this process (multi-pass)
    # or do a single pass in a specific order.
    # A single pass is O(|selected| * |ball|).
    # |selected| ~ 400. |ball| = 17. 400 * 17 = 6800 operations. Very fast.
    # We can do multiple passes until no more removals or time runs out.
    
    removed_any = True
    while removed_any:
        removed_any = False
        elapsed = time.time() - start_time
        if elapsed > deadline - 2:
            break
            
        # Shuffle to avoid order bias? Or just iterate.
        # Let's iterate in reverse order of selection (last added is easiest to remove if redundant).
        for i in range(len(selected) - 1, -1, -1):
            if elapsed > deadline - 2:
                break
            c = selected[i]
            can_remove = True
            for w in balls[c]:
                if word_cover_count[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                selected.pop(i)
                for w in balls[c]:
                    word_cover_count[w] -= 1
                removed_any = True
                # After removal, we should probably restart the loop or continue carefully.
                # Since removing c reduces counts for words in ball[c], it might make other removals impossible,
                # but it won't make other removals *easier* (counts only go down).
                # Wait, if we remove c, `word_cover_count` decreases.
                # If a word was covered by exactly 2 codewords, and one is removed, it's covered by 1.
                # This makes it *harder* to remove the other one.
                # So removals are "monotonic" in the sense that we can't create new removal opportunities.
                # Therefore, a single pass (or multiple passes if we added codewords in a way that creates redundancy)
                # is sufficient. But since we built the cover greedily, redundancy is likely.
                # However, once we remove a codeword, we never need to consider "re-adding" it.
                # So a single full pass through the list (in any order) is enough?
                # Actually, if we remove c, it affects w in ball[c].
                # Does it affect whether another codeword d (not in ball[c] intersection?) can be removed?
                # If d covers a word w not in ball[c], word_cover_count[w] is unchanged.
                # If d covers a word w in ball[c], word_cover_count[w] decreases.
                # This can only turn a "removable" state (count >= 2 for all w in ball[d]) into a "not removable" state (count == 1 for some w).
                # It cannot turn a "not removable" state into a "removable" one.
                # So, if we check codewords in a fixed order, and we remove one, we don't enable any subsequent removals that were previously blocked.
                # Thus, a single pass is sufficient to find all removable codewords?
                # Let's verify.
                # Suppose d was not removable because w_in_d had count 1.
                # If we remove c, and w_in_d is in ball[c], count becomes 0 (impossible, cover must be valid) or... wait.
                # If the cover is valid, min count is 1.
                # If w_in_d had count 1, it was covered ONLY by d.
                # If we remove c, and c covers w_in_d, then w_in_d count becomes 0. This would break the cover.
                # So if c covers w_in_d, and w_in_d count was 1, then c must be d.
                # So we can only remove c if it doesn't "break" any word.
                # If we successfully remove c, it means for all w in ball[c], count[w] >= 2.
                # After removal, count[w] becomes >= 1.
                # For any other codeword d, if it was not removable before, it means there exists w in ball[d] with count[w] == 1.
                # If w is not in ball[c], count[w] is unchanged, so d is still not removable.
                # If w is in ball[c], count[w] was >= 2 (since c was removable).
                # But we assumed count[w] == 1 for d to be not removable. Contradiction.
                # So if d was not removable, no w in ball[d] had count 1?
                # Wait. d is not removable if THERE EXISTS w in ball[d] with count[w] == 1.
                # If w in ball[d] had count 1, and w in ball[c], then count[w] must be >= 2 for c to be removable. Contradiction.
                # So w in ball[d] with count 1 CANNOT be in ball[c].
                # Therefore, removing c does not affect the "not removable" status of d.
                # Conclusion: A single pass through the list is sufficient to remove ALL possible redundant codewords.
                # We don't need the outer `while` loop, but it doesn't hurt (it will just break immediately).
                pass
            
    # Convert indices back to lists of integers
    result = []
    for idx in selected:
        result.append(list(all_words[idx]))
        
    return result

# EVOLVE-BLOCK-END
