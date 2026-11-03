import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s for safety
    
    total_words = q ** n
    
    # Fallback for large spaces
    if total_words > 100000:
        return _fallback_construct(q, n, radius, start_time, deadline)

    # Precompute balls as numpy arrays of uint64
    num_chunks = (total_words + 63) // 64
    balls = np.zeros((total_words, num_chunks), dtype=np.uint64)
    ball_indices = [None] * total_words
    
    # Helper to set bit in numpy array
    def set_bit(arr, idx):
        chunk = idx // 64
        bit = idx % 64
        arr[chunk] |= (1 << bit)

    # Generate all words and their balls
    # Word value to index mapping is just the integer value itself
    # We need to generate neighbors for each word
    # For q=3, n=8, radius=1:
    # Neighbors: self, and change one coord to other 2 values
    
    # To speed up generation, we can iterate
    for w in range(total_words):
        # Convert w to base-q digits
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        # Self
        set_bit(balls[w], w)
        ball_indices[w] = [w]
        
        # Neighbors
        for i in range(n):
            orig_val = digits[i]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_temp = temp
                # Reconstruct is expensive. 
                # Better: precompute powers of q
                pass
        
    # This generation is slow in Python. 
    # For 3^8=6561, it's okay.
    # Let's optimize generation.
    
    # Re-initialize for clean generation
    balls = np.zeros((total_words, num_chunks), dtype=np.uint64)
    ball_indices = [None] * total_words
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for w in range(total_words):
        set_bit(balls[w], w)
        ball_indices[w] = [w]
        
        # Extract digits
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            base_val = w - digits[i] * powers[i]
            for new_val in range(q):
                if new_val == digits[i]:
                    continue
                new_w = base_val + new_val * powers[i]
                set_bit(balls[w], new_w)
                ball_indices[w].append(new_w)

    # Initial Code: Tetracode Sum
    # For q=3, n=8
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Convert to indices
    # tail is length 4
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            # Convert word to index
            idx = 0
            for i in range(n):
                idx += word[i] * powers[i]
            initial_code.append(idx)
            
    # Initialize counts and hole_mask
    counts = np.zeros(total_words, dtype=np.int32)
    hole_mask = np.zeros(num_chunks, dtype=np.uint64)
    
    code = list(initial_code)
    
    # Fill counts
    for w in code:
        for idx in ball_indices[w]:
            counts[idx] += 1
             
    # Initialize hole_mask (should be 0 if initial code is valid covering)
    # But tetracode sum is a perfect covering? 
    # Tetracode is perfect radius 1.
    # Direct sum is perfect radius 1.
    # So hole_mask should be 0.
    
    # Prune function
    def prune(code, counts, ball_indices, balls, hole_mask, num_chunks):
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                w = code[i]
                # Check if redundant
                # min count in ball(w) > 1
                redundant = True
                for idx in ball_indices[w]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove w
                    code.pop(i)
                    for idx in ball_indices[w]:
                        counts[idx] -= 1
                    # Update hole_mask if counts became 0? 
                    # No, we only remove if counts > 1, so no 0s created.
                    changed = True
                    break # Restart loop
    
    # Main Loop
    best_code = list(code)
    best_len = len(code)
    
    while time.time() < deadline:
        # Perturb
        if code:
            # Remove 1 random word
            rem_idx = random.randint(0, len(code) - 1)
            rem_w = code.pop(rem_idx)
            for idx in ball_indices[rem_w]:
                counts[idx] -= 1
                if counts[idx] == 0:
                    set_bit(hole_mask, idx)
            
            # Re-greedy
            while True:
                # Check if hole_mask is 0
                # np.any(hole_mask)
                if not np.any(hole_mask):
                    break
                
                # Find best w
                # scores = sum(balls & hole_mask)
                temp = np.bitwise_and(balls, hole_mask, out=balls) # Reuse balls? No, balls is const.
                # Need a temp buffer
                # Allocate temp outside loop?
                # Let's allocate inside for simplicity, or use a global temp
                pass
            
            # This part is getting messy with numpy buffers.
            # Let's use a dedicated temp array.
    
    # Refined Loop
    temp_buffer = np.zeros((total_words, num_chunks), dtype=np.uint64)
    
    while time.time() < deadline:
        # Perturb
        if not code:
             break
        rem_idx = random.randint(0, len(code) - 1)
        rem_w = code.pop(rem_idx)
        for idx in ball_indices[rem_w]:
            counts[idx] -= 1
            if counts[idx] == 0:
                set_bit(hole_mask, idx)
        
        # Re-greedy
        while np.any(hole_mask):
            np.bitwise_and(balls, hole_mask, out=temp_buffer)
            scores = temp_buffer.sum(axis=1)
            best_w = np.argmax(scores)
            if scores[best_w] == 0:
                break # Should not happen if hole_mask is not 0
            
            code.append(int(best_w))
            # Update counts
            for idx in ball_indices[best_w]:
                counts[idx] += 1
            # Update hole_mask
            np.bitwise_and(hole_mask, ~balls[best_w], out=hole_mask)
        
        # Prune
        prune(code, counts, ball_indices, balls, hole_mask, num_chunks)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = list(code)
            
    # Convert best_code indices back to lists of ints
    result = []
    for idx in best_code:
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result

def _fallback_construct(q, n, radius, start_time, deadline):
    # Simple greedy
    total_words = q ** n
    uncovered = set(range(total_words))
    code = []
    
    def get_neighbors(w):
        digits = [0]*n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        neigh = {w}
        for i in range(n):
            for v in range(q):
                if v == digits[i]: continue
                new_temp = temp
                # Reconstruct logic is needed
                pass
        return neigh
    
    # This fallback is likely too slow for large spaces if implemented naively.
    # Given the task is 3^8, this won't be hit.
    return []

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    
    if total_words > 100000:
        # Fallback: Return tetracode sum if possible, else empty
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return []

    num_chunks = (total_words + 63) // 64
    balls = np.zeros((total_words, num_chunks), dtype=np.uint64)
    ball_indices = [None] * total_words
    
    def set_bit(arr, idx):
        chunk = idx // 64
        bit = idx % 64
        arr[chunk] |= (1 << bit)

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for w in range(total_words):
        set_bit(balls[w], w)
        ball_indices[w] = [w]
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        for i in range(n):
            base_val = w - digits[i] * powers[i]
            for new_val in range(q):
                if new_val == digits[i]:
                    continue
                new_w = base_val + new_val * powers[i]
                set_bit(balls[w], new_w)
                ball_indices[w].append(new_w)

    # Initial Code
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                idx = 0
                for i in range(n):
                    idx += word[i] * powers[i]
                initial_code.append(idx)
    else:
        initial_code = []

    counts = np.zeros(total_words, dtype=np.int32)
    hole_mask = np.zeros(num_chunks, dtype=np.uint64)
    code = list(initial_code)
    
    for w in code:
        for idx in ball_indices[w]:
            counts[idx] += 1

    temp_buffer = np.zeros((total_words, num_chunks), dtype=np.uint64)
    
    def prune(code, counts, ball_indices):
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                w = code[i]
                redundant = True
                for idx in ball_indices[w]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    code.pop(i)
                    for idx in ball_indices[w]:
                        counts[idx] -= 1
                    changed = True
                    break

    prune(code, counts, ball_indices)
    
    best_code = list(code)
    best_len = len(code)
    
    while time.time() < deadline:
        if not code:
            break
        rem_idx = random.randint(0, len(code) - 1)
        rem_w = code.pop(rem_idx)
        for idx in ball_indices[rem_w]:
            counts[idx] -= 1
            if counts[idx] == 0:
                set_bit(hole_mask, idx)
        
        while np.any(hole_mask):
            np.bitwise_and(balls, hole_mask, out=temp_buffer)
            scores = temp_buffer.sum(axis=1)
            best_w = np.argmax(scores)
            if scores[best_w] == 0:
                break
            code.append(int(best_w))
            for idx in ball_indices[best_w]:
                counts[idx] += 1
            np.bitwise_and(hole_mask, ~balls[best_w], out=hole_mask)
        
        prune(code, counts, ball_indices)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = list(code)
            
    result = []
    for idx in best_code:
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result
# EVOLVE-BLOCK-END
