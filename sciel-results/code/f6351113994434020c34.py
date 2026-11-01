import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all words and their indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    total_words = q ** n
    
    # Precompute balls for all words
    # ball_idx[idx] = list of indices of words in ball(idx)
    # ball_set[idx] = set of indices for O(1) lookup
    ball_idx = [None] * total_words
    ball_set = [None] * total_words
    for w in all_words:
        idx = word_to_idx[w]
        b = [idx]
        for i in range(n):
            for val in range(q):
                if w[i] != val:
                    new_word = list(w)
                    new_word[i] = val
                    b.append(word_to_idx[tuple(new_word)])
        ball_idx[idx] = b
        ball_set[idx] = set(b)
        
    # 1. Generate initial codewords using Greedy Set Cover
    # This is generally better than product constructions for small n
    covered_mask = np.zeros(total_words, dtype=bool)
    cw_idxs = set()
    
    # Randomize order of words to pick for greedy to avoid bias, 
    # but standard greedy picks the word that covers the most *new* words.
    # To speed up, we can just iterate through all words and pick the best one repeatedly.
    # However, for 729 words, we can be a bit smarter.
    
    # Initialize: pick a random start
    first_idx = random.randint(0, total_words - 1)
    cw_idxs.add(first_idx)
    covered_mask[ball_idx[first_idx]] = True
    
    # Greedy loop
    while not covered_mask.all():
        if time.time() > deadline - 5:
            break # Safety: if greedy takes too long, stop and rely on local search? 
                  # Actually, greedy on 729 items is very fast.
        
        best_idx = -1
        max_new = 0
        
        # We can optimize this by only checking words that cover at least one new word.
        # But checking all 729 words is $729 \times 13 \approx 10k$ ops, which is tiny.
        for idx in range(total_words):
            if idx in cw_idxs:
                continue
            b = ball_idx[idx]
            # Count how many words in b are NOT covered
            # Using numpy boolean mask intersection
            new_count = np.sum(covered_mask[b] == False)
            if new_count > max_new:
                max_new = new_count
                best_idx = idx
                
        if best_idx == -1:
            # Should not happen if not all covered
            break
            
        cw_idxs.add(best_idx)
        covered_mask[ball_idx[best_idx]] = True
        
    # Convert to list for easier manipulation if needed, but set is fine for membership
    # Initialize cover_count
    cover_count = np.zeros(total_words, dtype=int)
    for cw in cw_idxs:
        cover_count[ball_idx[cw]] += 1
        
    # Helper functions
    def is_redundant(cw_idx):
        # A codeword is redundant if every word it covers is covered by at least one other codeword
        # i.e., for all w in ball(cw_idx), cover_count[w] > 1
        return np.all(cover_count[ball_idx[cw_idx]] > 1)
        
    def remove_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] -= 1
        cw_idxs.remove(cw_idx)
        
    def add_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] += 1
        cw_idxs.add(cw_idx)
            
    # Main Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant codewords
        redundant_found = True
        while redundant_found and time.time() < deadline:
            redundant_found = False
            cws_to_check = list(cw_idxs)
            random.shuffle(cws_to_check)
            for cw in cws_to_check:
                if is_redundant(cw):
                    remove_cw(cw)
                    improved = True
                    redundant_found = True
                    break # Restart check after removal
        
        # Phase 2: Try 2-for-1 replacements (Remove 2, Add 1)
        # This allows reducing the code size by 1 if two codewords' coverage can be merged into one.
        if not improved and time.time() < deadline:
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                if len(cw_idxs) < 2:
                    break
                
                # Pick two random distinct codewords
                c1, c2 = random.sample(list(cw_idxs), 2)
                
                # Temporarily remove them
                remove_cw(c1)
                remove_cw(c2)
                
                # Find words that are now uncovered (cover_count == 0)
                # These must be covered by the new candidate
                uncovered_mask = (cover_count == 0)
                
                if not np.any(uncovered_mask):
                    # Both were redundant? Should have been caught in Phase 1.
                    # But maybe they were jointly redundant?
                    # If no uncovered words, we have reduced size by 2.
                    improved = True
                    break
                else:
                    # We need to find a word w such that ball(w) covers all uncovered words.
                    # i.e., uncovered_words subset of ball(w)
                    # This implies uncovered_words must be a "ball" or subset of a ball.
                    # Specifically, w must be within distance 1 of all uncovered words.
                    
                    # Optimization: 
                    # If there are many uncovered words, it's unlikely a single w covers them all.
                    # The max number of words covered by one ball is 1 + n*(q-1).
                    # For q=3, n=6, ball size is 1 + 6*2 = 13.
                    # So if > 13 words are uncovered, it's impossible.
                    num_uncovered = np.sum(uncovered_mask)
                    
                    if num_uncovered > 1 + n * (q - 1):
                        # Impossible to cover with 1 word
                        add_cw(c1)
                        add_cw(c2)
                        continue
                    
                    # Get the list of uncovered indices
                    uncovered_list = np.where(uncovered_mask)[0]
                    
                    # Candidate w must be in the intersection of balls of all uncovered words.
                    # Intersection of balls = { w | d(w, u) <= 1 for all u in uncovered_list }
                    
                    # We can iterate through potential candidates.
                    # A good set of candidates is the ball of the first uncovered word.
                    u0 = uncovered_list[0]
                    candidates = ball_idx[u0]
                    
                    valid_candidate = -1
                    for w_cand in candidates:
                        # Check if w_cand covers all uncovered words
                        # i.e., all(u in ball_set[w_cand] for u in uncovered_list)
                        # Using sets is fast.
                        if all(u in ball_set[w_cand] for u in uncovered_list):
                            valid_candidate = w_cand
                            break
                    
                    if valid_candidate != -1:
                        # Add the new word
                        add_cw(valid_candidate)
                        improved = True
                        break
                    else:
                        # Revert
                        add_cw(c1)
                        add_cw(c2)
                        
        # Phase 3: Try 1-for-1 swaps (Keep size, change configuration)
        # This helps escape local minima where no 2-for-1 move is possible, but a swap
        # enables a future 2-for-1 move.
        if not improved and time.time() < deadline:
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                if not cw_idxs:
                    break
                c_to_swap = random.choice(list(cw_idxs))
                
                # Temporarily remove
                remove_cw(c_to_swap)
                
                # Find uncovered
                uncovered_mask = (cover_count == 0)
                if not np.any(uncovered_mask):
                    # Redundant, should be caught in Phase 1
                    improved = True
                    break
                else:
                    num_uncovered = np.sum(uncovered_mask)
                    if num_uncovered > 1 + n * (q - 1):
                        add_cw(c_to_swap)
                        continue
                        
                    uncovered_list = np.where(uncovered_mask)[0]
                    u0 = uncovered_list[0]
                    candidates = ball_idx[u0]
                    
                    valid_candidate = -1
                    for w_cand in candidates:
                        if w_cand == c_to_swap:
                            continue
                        if all(u in ball_set[w_cand] for u in uncovered_list):
                            valid_candidate = w_cand
                            break
                    
                    if valid_candidate != -1:
                        add_cw(valid_candidate)
                        improved = True
                        break
                    else:
                        add_cw(c_to_swap)

    # Convert back to list of lists
    result = [list(idx_to_word[idx]) for idx in cw_idxs]
    
    return result
