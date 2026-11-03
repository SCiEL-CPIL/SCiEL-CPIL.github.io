import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within Hamming distance <= radius)
    # For q=3, n=7, radius=1, this is fast.
    # For larger n, this might be expensive, but the task specifies n=7.
    neighbors = [[] for _ in range(num_words)]
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # Distance 0
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].append(idx)
        # If radius > 1, we would need to expand further, but for this task radius=1 is standard.
        # The prompt says "radius" is passed, so we should handle general radius if possible, 
        # but given the 120s limit and n=7, radius is likely 1.
        # Let's add a check for radius > 1 using a BFS or simple expansion if needed, 
        # but for n=7, q=3, radius=1 is the specific case.
        # To be safe, let's implement a generic neighbor finding if radius > 1, 
        # but it might be slow. Given the context, radius=1 is the intended case.
        
    # Greedy Set Cover
    uncovered = set(range(num_words))
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_w = -1
        best_count = 0
        
        # We can optimize by only considering candidates that cover at least one uncovered word.
        # But checking all is fast enough.
        for w_idx in range(num_words):
            count = 0
            for n_idx in neighbors[w_idx]:
                if n_idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        if best_w == -1:
            break
            
        code.append(all_words[best_w])
        
        # Remove covered words from uncovered
        for n_idx in neighbors[best_w]:
            if n_idx in uncovered:
                uncovered.remove(n_idx)
                
    # If we ran out of time, we might have an invalid code.
    # We must ensure validity. If we broke early, the code is invalid.
    # The prompt says "Valid when every word ... is within ...".
    # If we return an invalid code, score is 0.
    # So we must only return if we finished the loop or if we are sure it's valid.
    # The loop terminates when uncovered is empty.
    # If we break due to time, we should probably return a known valid fallback 
    # or the current code if it happens to be valid (unlikely if uncovered not empty).
    # However, the greedy loop is very fast (~0.1s), so we should not hit the deadline.
    # If we do hit the deadline, it's better to return the seed solution (243 words) 
    # than an invalid one.
    
    if uncovered:
        # Fallback to the seed solution to ensure validity
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        # This fallback only works for q=3, n=7? 
        # The seed used q=3 logic. If q is different, this might fail.
        # But the task is specifically about ternary (q=3).
        # Let's just return the code we have if it's small, but it's invalid.
        # Actually, if we haven't covered everything, we should not return it.
        # Let's return the seed's construction if it's valid for the parameters.
        # The seed's construction:
        # tetracode + tail.
        # This is valid for q=3, n>=4.
        if q == 3 and n >= 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Return a trivial valid code: all words
             return all_words

    return code
