import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball indices for each word
    # balls[i] is a list of indices of words within 'radius' of all_words[i]
    balls = []
    
    # Helper to calculate Hamming distance efficiently for small n
    # Since n=6 and q=3, simple loop is fine, but we can optimize slightly
    def hamming_dist(i, j):
        w1 = all_words[i]
        w2 = all_words[j]
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    # Precomputation step
    for i in range(total_words):
        ball = []
        for j in range(total_words):
            if hamming_dist(i, j) <= radius:
                ball.append(j)
        balls.append(ball)
        
    # Initialize code with symmetry breaking: first word is all zeros
    first_idx = word_to_idx[(0,) * n]
    code = [first_idx]
    
    # Coverage counts: how many codewords cover each word
    cover_count = [0] * total_words
    for x in balls[first_idx]:
        cover_count[x] += 1
        
    # Greedy Phase
    # Uncovered set as a bitset for quick check? 
    # Actually, maintaining cover_count is easier for the local search phase.
    # For greedy, we just need to find the word that covers the most currently uncovered words.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find best word to add
        best_idx = -1
        best_gain = 0
        
        # We can iterate through all words. 
        # Optimization: Only consider words that cover at least one uncovered word.
        # An uncovered word is one where cover_count[x] == 0.
        
        # To speed up, we can just check all words. 729 words is small.
        for i in range(total_words):
            gain = 0
            for x in balls[i]:
                if cover_count[x] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            # All words are covered
            break
            
        # Add the best word
        code.append(best_idx)
        for x in balls[best_idx]:
            cover_count[x] += 1
            
    # Local Search Phase
    # Strategy: 
    # 1. Try to remove any redundant codeword.
    # 2. If no removal, try to swap a codeword with a different one to enable future removals or just improve "quality" (though size is the metric, so only removals reduce size).
    #    Actually, swaps don't reduce size directly. But a swap might make a *different* codeword redundant.
    #    So:
    #    Loop:
    #      - Try Remove: If any c is redundant, remove it.
    #      - If no remove, Try Swap: 
    #         For each c in code:
    #           Try swapping c with w (w not in code) such that the new code is valid.
    #           If a valid swap is found, do it.
    #         If no swap found, break (local optimum).
    #      - (Optional) Perturbation if stuck, but let's stick to simple local search first.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Remove
        removed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if c is redundant
            redundant = True
            for x in balls[c]:
                if cover_count[x] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                for x in balls[c]:
                    cover_count[x] -= 1
                code.pop(i)
                removed = True
                improved = True
                # Restart the loop to check for new redundancies
                break
        
        if removed:
            continue
            
        # 2. Try Swap
        # We want to find a swap c -> w such that the new code is valid.
        # This is equivalent to: (Code - {c} + {w}) is a valid cover.
        # Condition: For all x, cover_count[x] - [x in Ball(c)] + [x in Ball(w)] > 0.
        # This is hard to check directly for all x quickly without updating.
        # Instead, let's just simulate the swap for candidates.
        
        swapped = False
        # Iterate through codewords
        for i in range(len(code)):
            c = code[i]
            
            # Candidate w: any word not in code.
            # To be efficient, we can pick w that covers "many" words with cover_count == 1 
            # that are currently covered by c? 
            # Or just try a few random w's? 
            # Given the small size, we can try all w? 
            # 729 * 73 * (cost of validity check).
            # Validity check: 
            #   Temporarily remove c (decrement counts).
            #   Check if any count == 0. If yes, c is critical.
            #   If c is critical, we need w to cover those 0-count words.
            #   Then add w (increment counts).
            #   Check if any count == 0. If no, valid.
            
            # Let's identify "critical" words for c: words x in Ball(c) with cover_count[x] == 1.
            critical_words = []
            for x in balls[c]:
                if cover_count[x] == 1:
                    critical_words.append(x)
                    
            if not critical_words:
                # c is redundant? We should have caught this in Remove phase.
                # But maybe cover_count changed? No, we only change on remove/swap.
                # So if no critical words, c is redundant.
                # Why didn't Remove catch it?
                # Ah, Remove loop breaks after first removal. 
                # So if we are in Swap phase, no codeword is redundant.
                # Thus critical_words must be non-empty for all c.
                pass
            
            # We need to find w such that w covers ALL critical_words.
            # Because if we remove c, critical_words become 0. 
            # So w must cover them.
            # Additionally, w must not "break" anything else? 
            # If w covers critical_words, they become >= 1.
            # What about words not in Ball(c)? They are unaffected by removing c.
            # What about words in Ball(c) but not critical (count >= 2)? 
            # Removing c makes them >= 1. Adding w might make them higher.
            # So the ONLY condition for validity of swap c->w is:
            # w must cover all x in Ball(c) such that cover_count[x] == 1.
            
            # So, find w not in code that covers all critical_words.
            # If found, swap.
            
            # Optimization: Precompute which words cover a specific set? 
            # Or just iterate w?
            
            # Let's iterate w
            for w in range(total_words):
                if w in code:
                    continue
                    
                # Check if w covers all critical_words
                valid_swap = True
                for x in critical_words:
                    if x not in balls[w]:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Perform Swap
                    # Remove c
                    for x in balls[c]:
                        cover_count[x] -= 1
                    # Add w
                    for x in balls[w]:
                        cover_count[x] += 1
                    
                    code[i] = w
                    swapped = True
                    improved = True
                    break
            
            if swapped:
                break
                
    return [list(all_words[i]) for i in code]
