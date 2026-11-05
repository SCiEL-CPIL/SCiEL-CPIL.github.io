import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    # Generate all words
    # Map word (tuple) to index
    # q=3, n=6, total words = 729
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Pre-compute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1, each word has 1 + 1*(n*(q-1)) = 1 + 6*2 = 13 neighbors? 
    # Wait, distance 1 means changing exactly one position.
    # Number of neighbors at dist 1: n * (q-1) = 6 * 2 = 12.
    # Plus the word itself (dist 0). Total 13.
    
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    neighbors[i].append(word_to_idx[new_w])
                    
    # Greedy Set Cover
    # State: covered (bool array)
    # We want to minimize the number of codewords.
    # At each step, pick the word that covers the most *uncovered* words.
    
    covered = [False] * num_words
    code_indices = []
    
    # Initial heuristic: Start with a random word? Or 0?
    # Random restarts can help escape local minima if we do local search, 
    # but for Set Cover, Greedy is usually strong.
    # Let's just run Greedy.
    
    # To speed up, we can maintain a count of uncovered neighbors for each candidate.
    # But recalculating is cheap enough for 729 words.
    
    while True:
        if time.time() > deadline:
            break
            
        if all(covered):
            break
            
        best_idx = -1
        max_new = -1
        
        # Optimization: Only consider candidates that cover at least one uncovered word.
        # We can iterate through all candidates.
        for i in range(num_words):
            # Quick check: if all neighbors of i are covered, skip.
            # But we need to count how many are NOT covered.
            new_covered = 0
            for nb in neighbors[i]:
                if not covered[nb]:
                    new_covered += 1
                    # Early exit if we can't beat max_new? 
                    # No, we need the exact count to compare.
                    # But if new_covered == 0, break early.
            
            if new_covered > max_new:
                max_new = new_covered
                best_idx = i
                
                # Early termination for inner loop? 
                # If max_new is very high, we might still need to check others.
                # But if max_new == len(neighbors[i]) (all uncovered), that's the max possible.
                # Max possible new covered is len(neighbors[i]) which is constant (13).
                # So if max_new == 13, we can break the outer candidate loop?
                # No, because another candidate might also cover 13, but we just need one.
                # Actually, if we find a candidate covering 13 new words, it's optimal for this step.
                # We can break the candidate loop.
                if max_new == len(neighbors[0]): 
                    break
        
        if best_idx == -1:
            # Should not happen if not all covered
            break
            
        code_indices.append(best_idx)
        
        # Update covered status
        for nb in neighbors[best_idx]:
            covered[nb] = True
            
    # If we finished the loop successfully (all covered), return the code
    if all(covered):
        # Convert indices back to word lists
        result = [list(all_words[i]) for i in code_indices]
        # Ensure distinctness (guaranteed by greedy picking distinct centers? 
        # Yes, we pick a center, mark its neighbors covered. 
        # Can we pick the same center again? 
        # If we pick center C, all its neighbors are covered. 
        # In the next step, C's neighbors are all covered, so new_covered for C will be 0.
        # Since there are uncovered words, some other candidate must have new_covered > 0.
        # So we won't pick C again.
        return result
        
    # Fallback: If greedy didn't finish in time or failed, use the seed construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
