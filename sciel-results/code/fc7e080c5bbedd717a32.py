import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute powers of q for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def idx_to_word(idx):
        w = [0] * n
        for k in range(n):
            w[k] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for k in range(n):
            idx += w[k] * powers[k]
        return idx

    # Precompute spheres as bitsets
    spheres = [0] * total_words
    
    for i in range(total_words):
        w = idx_to_word(i)
        spheres[i] |= (1 << i)
        
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        j = word_to_idx(w)
                        spheres[i] |= (1 << j)
                w[pos] = original_val

        if time.time() > deadline:
            break

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code_indices = []
    
    # Optimization: We can skip candidates that are already covered? 
    # No, a covered candidate might still cover other uncovered points.
    # However, if a candidate's sphere is fully contained in the union of previous spheres, it's useless.
    # The greedy step handles this naturally by selecting max marginal gain.
    
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            count = (spheres[i] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        uncovered &= ~spheres[best_idx]
        
        if time.time() > deadline:
            break

    code_words = [idx_to_word(i) for i in code_indices]
    
    return code_words
