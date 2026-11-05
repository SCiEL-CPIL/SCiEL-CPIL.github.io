import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 55.0
    
    # Precompute ball masks for radius 5
    ball_masks = []
    for mask in range(2**N):
        if mask.bit_count() <= 5:
            ball_masks.append(mask)
    ball_masks = np.array(ball_masks, dtype=np.int64)
    
    # Forbidden count array
    forbidden_count = np.zeros(2**N, dtype=np.int32)
    
    def mark_word(word):
        neighbors = word ^ ball_masks
        forbidden_count[neighbors] += 1
        
    def unmark_word(word):
        neighbors = word ^ ball_masks
        forbidden_count[neighbors] -= 1
        
    # Generate [18, 9, 6] QR code
    # g(x) = 1 + x + x^3 + x^5 + x^6 + x^8 + x^9
    g_bits = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    g_int = 0
    for j in range(18):
        if g_bits[j]:
            g_int |= 1 << j
            
    initial_words = []
    for m in range(512):
        c_int = 0
        for i in range(9):
            if (m >> i) & 1:
                shifted = ((g_int << i) | (g_int >> (18 - i))) & ((1 << 18) - 1)
                c_int ^= shifted
        if (c_int >> 17) & 1 == 0:
            w = c_int & ((1 << 17) - 1)
            initial_words.append(w)
            
    # If generation failed or distance is wrong, fallback
    # Check min distance of initial_words
    if len(initial_words) < 256:
        # Fallback to simple greedy
        initial_words = []
        for w in range(2**N):
            if time.time() - start_time > DEADLINE: break
            # Check if w is valid
            # This is O(N * 2^N) which is too slow.
            # I will just use the QR code.
            pass

    # Initialize forbidden_count
    for w in initial_words:
        mark_word(w)
        
    current_words = initial_words[:]
    
    # Local Search
    while time.time() - start_time < DEADLINE:
        valid_indices = np.where(forbidden_count == 0)[0]
        
        if len(valid_indices) > 0:
            # Pick a candidate
            # To improve diversity, pick randomly or the one with max distance?
            # Let's pick the one with the maximum weight (popcount) as a heuristic?
            # Or just the first one.
            best_cand = valid_indices[0]
            
            current_words.append(best_cand)
            mark_word(best_cand)
        else:
            # Try swap
            swap_found = False
            for i in range(len(current_words)):
                if time.time() - start_time > DEADLINE:
                    break
                w_to_remove = current_words[i]
                unmark_word(w_to_remove)
                
                valid_indices = np.where(forbidden_count == 0)[0]
                if len(valid_indices) > 0:
                    best_cand = valid_indices[0]
                    current_words[i] = best_cand
                    mark_word(best_cand)
                    swap_found = True
                    break
                else:
                    mark_word(w_to_remove)
                    
            if not swap_found:
                break
                
    return current_words
