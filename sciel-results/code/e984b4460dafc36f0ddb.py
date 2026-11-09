"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*2 = 15
    
    # Fallback for other parameters (not expected in this specific evaluation)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Precompute powers of q for index conversion
    q_pows = [1]
    for i in range(n):
        q_pows.append(q_pows[-1] * q)
    # q_pows[i] = q^i

    # Precompute the ball for each word as a Python integer bitmask
    # bit j is set in ball_masks[i] if word j is within distance radius of word i
    ball_masks = [0] * total_words
    for i in range(total_words):
        mask = 1 << i
        # For each position
        for pos in range(n):
            current_digit = (i // q_pows[pos]) % q
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                diff = new_digit - current_digit
                new_idx = i + diff * q_pows[pos]
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1

    # Convert an index to a list of digits
    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        return word  # Note: word[0] is LSB, word[n-1] is MSB

    best_code = None
    best_size = float('inf')

    # Use the seed solution as a starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Convert seed words to indices
    seed_indices = []
    for w in seed_words:
        # w is in MSB-first order (word[0] is most significant)
        # But idx_to_word gives LSB-first
        # So we need to reverse
        idx = 0
        for k in range(n):
            idx += w[k] * (q_pows[n - 1 - k])
        seed_indices.append(idx)
    best_code = seed_indices[:]
    best_size = len(seed_indices)

    # Randomized Greedy Set Cover with multiple restarts
    # We start from the seed code and try to improve by removing words or replacing them
    # Actually, a simpler approach: run greedy from scratch multiple times
    
    # To make it faster, we can use a priority queue or just iterate.
    # Since 2187 is small, iterating is fine.
    
    # Optimization: Precompute the list of candidate indices
    candidates = list(range(total_words))
    
    # We will run several rounds of greedy.
    # In each round, we pick the word that covers the most uncovered points.
    
    # To speed up, we can use a trick:
    # Instead of checking all 2187 words every time, we can maintain a list of "active" candidates.
    # But for 2187, it's fast enough.
    
    # Let's do multiple restarts.
    # Each restart:
    # 1. Initialize uncovered = full_mask
    # 2. While uncovered:
    #    a. Find the word that covers the most uncovered points.
    #    b. Add it to the code.
    #    c. Remove covered points from uncovered.
    # 3. If the code size is better, update best.
    
    # To add diversity, we can shuffle the candidates or pick randomly among ties.
    
    # Let's also try a "local search" improvement on the best code found.
    # But for now, just greedy restarts.
    
    # Number of restarts: we will run until deadline.
    # Each greedy run takes O(total_words * code_size) bit operations.
    # code_size is ~200, total_words is 2187.
    # 2187 * 200 * (cost of bit_count) is fast.
    # bit_count on 2187-bit integers is very fast in Python 3.8+.
    
    # Let's estimate: 2187 * 200 = 437,400 bit_count operations per greedy run.
    # Each bit_count is ~100ns? So ~0.04s per run.
    # We can do ~2000 runs in 110s. That's plenty.
    
    # Let's run greedy restarts.
    
    while time.time() < deadline:
        uncovered = full_mask
        code = []
        
        while uncovered:
            max_count = -1
            best_candidates = []
            
            # Find the word that covers the most uncovered points
            for c in candidates:
                count = (ball_masks[c] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    best_candidates = [c]
                elif count == max_count:
                    best_candidates.append(c)
            
            # Pick one randomly from the best candidates
            if best_candidates:
                chosen = random.choice(best_candidates)
            else:
                break  # Should not happen
            
            code.append(chosen)
            uncovered &= ~ball_masks[chosen]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we reach the known best (186), we can stop or continue to see if we find better.
            # Let's continue to see if we find 185 or lower.
            # But if we are close to deadline, we should stop.
            if best_size <= 186 and time.time() > deadline - 10:
                break

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    
    return result
# EVOLVE-BLOCK-END
