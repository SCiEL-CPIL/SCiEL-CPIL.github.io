import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145
    
    total_words = q ** n
    
    # If space is too large for bitmask precomputation, use a fallback
    # 2^15 = 32768 bits is fine for Python int.
    # But O(total_words^2) precomputation is the bottleneck.
    # 32768^2 is ~10^9, too slow.
    # Let's set a threshold.
    if total_words > 5000:
        # Fallback: Simple greedy on the fly
        # This might be slow but better than nothing for large spaces
        # Or just return a random set? No, must be valid.
        # For now, assume small space as per task description.
        # If large, we try a simple greedy without precomputation.
        
        all_words = []
        for i in range(total_words):
            w = [0] * n
            idx = i
            for j in range(n - 1, -1, -1):
                w[j] = idx % q
                idx //= q
            all_words.append(w)
        
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            best_candidate = -1
            max_gain = -1
            # Sample candidates if too many
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c_idx in candidates:
                if c_idx in code: # Using list for 'in' is O(N), use set
                    continue
                # Calculate gain
                gain = 0
                for u_idx in uncovered:
                    if hamming_dist(all_words[c_idx], all_words[u_idx]) <= radius:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_candidate = c_idx
            
            if best_candidate == -1:
                break
                
            code.append(best_candidate)
            # Update uncovered
            new_uncovered = set()
            for u_idx in uncovered:
                if hamming_dist(all_words[best_candidate], all_words[u_idx]) > radius:
                    new_uncovered.add(u_idx)
            uncovered = new_uncovered
            
            if time.time() > deadline:
                break
                
        return [all_words[i] for i in code]

    # Small space: Bitmask approach
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(total_words)]
    
    balls = [0] * total_words
    
    for i in range(total_words):
        center = all_words[i]
        mask = 0
        for j in range(total_words):
            w = all_words[j]
            dist = sum(1 for a, b in zip(center, w) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        balls[i] = mask

    target_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words + 1
    
    num_attempts = 1000
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
            
        uncovered = target_mask
        code = []
        used = [False] * total_words
        
        while uncovered:
            max_gain = -1
            best_candidates = []
            
            for i in range(total_words):
                if used[i]:
                    continue
                new_covered = balls[i] & uncovered
                gain = new_covered.bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            used[chosen] = True
            uncovered &= ~balls[chosen]
            
            if max_gain == 0:
                break

        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # Redundancy removal
            changed = True
            while changed:
                changed = False
                for idx_in_code, c in enumerate(best_code):
                    union_others = 0
                    for other_idx, other_c in enumerate(best_code):
                        if other_idx == idx_in_code:
                            continue
                        union_others |= balls[other_c]
                    
                    if (balls[c] & ~union_others) == 0:
                        best_code.pop(idx_in_code)
                        changed = True
                        break
            
            if len(best_code) < best_size:
                best_size = len(best_code)

    if best_code is None:
        # Fallback to seed logic if valid
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [all_words[i] for i in range(total_words)] # Worst case: all words

    return [all_words[i] for i in best_code]
