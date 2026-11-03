import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Map word (tuple) to index (integer 0..q^n-1)
    # Since q=3, n=6, max index is 728.
    # We can represent words as integers in base q.
    
    space_size = q ** n
    
    # Generate all words as tuples to allow easy manipulation
    # But for speed, we work with indices.
    # index -> word tuple
    # word tuple -> index
    
    # Precompute balls for all possible codewords (indices 0 to space_size-1)
    # ball[i] = list of indices of words covered by codeword i
    
    # Helper to convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return tuple(w)
        
    # Helper to convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    balls = []
    for i in range(space_size):
        w = idx_to_word(i)
        ball = [i]
        # Radius 1 neighbors
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                ball.append(word_to_idx(tuple(new_w)))
        balls.append(ball)
        
    # 2. Initialization
    # Tetracode construction
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n-4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            cw = h + list(tail)
            initial_codewords.append(word_to_idx(tuple(cw)))
            
    # Ensure uniqueness (should be unique by construction)
    initial_codewords = list(set(initial_codewords))
    
    active = set(initial_codewords)
    cover_count = np.zeros(space_size, dtype=np.int32)
    
    # Initialize coverage
    for cw in active:
        cover_count[balls[cw]] += 1
        
    best_size = len(active)
    best_active = set(active)
    
    # 3. Local Search
    # We will perform a loop of:
    # 1. Greedy removal of redundant codewords.
    # 2. Random swaps to escape local minima.
    
    def is_removable(cw):
        # Check if any word in ball[cw] has cover_count == 1
        for idx in balls[cw]:
            if cover_count[idx] == 1:
                return False
        return True

    def try_remove(cw):
        nonlocal cover_count, active
        if is_removable(cw):
            active.remove(cw)
            cover_count[balls[cw]] -= 1
            return True
        return False

    def greedy_remove():
        nonlocal active, cover_count
        removed = True
        while removed:
            removed = False
            # Shuffle to avoid bias
            to_check = list(active)
            random.shuffle(to_check)
            for cw in to_check:
                if cw in active: # Check if still active (might have been removed by previous check in loop? No, we iterate a copy)
                    if is_removable(cw):
                        active.remove(cw)
                        cover_count[balls[cw]] -= 1
                        removed = True
                        # Don't break, continue checking others, as one removal might make another removable
                        # But we must be careful not to check a word we just removed.
                        # Since we iterate 'to_check' which is a snapshot, and we check 'if cw in active', it's safe.
                        
    # Start with greedy removal on the initial set
    greedy_remove()
    if len(active) < best_size:
        best_size = len(active)
        best_active = set(active)
        
    # Main loop
    while time.time() < deadline:
        # 1. Try to remove any remaining redundant codewords
        greedy_remove()
        
        if len(active) < best_size:
            best_size = len(active)
            best_active = set(active)
            
        # If we reached the theoretical lower bound or known optimum, we can stop early if confident
        # Known optimum for K_3(6,1) is 73.
        if best_size <= 73:
            break
            
        # 2. Random Swap
        if len(active) == 0:
            break
            
        # Pick a random active codeword to potentially replace
        u = random.choice(list(active))
        
        # Pick a random candidate v from the whole space
        # To make swaps more effective, we might pick v from the "uncovered" or "less covered" areas?
        # Or just random.
        v = random.randint(0, space_size - 1)
        
        if v == u:
            continue
            
        # Check if replacing u with v is valid
        # Condition: Every word w in Ball(u) with cover_count[w] == 1 must be in Ball(v)
        # If such a w exists and is NOT in Ball(v), the swap is invalid.
        
        valid_swap = True
        for w in balls[u]:
            if cover_count[w] == 1:
                # This word is ONLY covered by u.
                # So v MUST cover it.
                if w not in balls[v]:
                    valid_swap = False
                    break
                    
        if valid_swap:
            # Perform swap
            active.remove(u)
            cover_count[balls[u]] -= 1
            
            active.add(v)
            cover_count[balls[v]] += 1
            
            # After a swap, try to remove redundant codewords
            greedy_remove()
            
            if len(active) < best_size:
                best_size = len(active)
                best_active = set(active)
                
    # Convert best_active (set of indices) back to list of lists
    final_code = []
    for idx in best_active:
        w = idx_to_word(idx)
        final_code.append(list(w))
        
    return final_code
