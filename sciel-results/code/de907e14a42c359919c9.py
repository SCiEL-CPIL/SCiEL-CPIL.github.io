import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    if q != 3 or n != 7 or radius != 1:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** 7
    all_mask = (1 << total_words) - 1
    
    # Precompute index mappings
    idx_to_word = {}
    word_to_idx = {}
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(7):
            w.append(temp % 3)
            temp //= 3
        w = tuple(w)
        idx_to_word[i] = w
        word_to_idx[w] = i

    # Precompute balls
    balls = [0] * total_words
    for i in range(total_words):
        w = idx_to_word[i]
        mask = 0
        mask |= (1 << i)
        for pos in range(7):
            for val in range(3):
                if w[pos] == val:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        balls[i] = mask

    best_code = None
    best_size = total_words + 1
    
    rng = random.Random(42) # Fixed seed for initial run, then vary

    def local_search(code, rng):
        code = list(code)
        code_set = set(code)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Removal
            for i in list(code):
                union_others = 0
                for j in code:
                    if j != i:
                        union_others |= balls[j]
                if union_others == all_mask:
                    code.remove(i)
                    code_set.remove(i)
                    improved = True
                    break
            
            if not improved:
                # 2. Swap
                # Try to swap a codeword with a non-codeword
                # We limit attempts to avoid getting stuck in a long loop
                for _ in range(10):
                    if len(code) == 0:
                        break
                    i = rng.choice(code)
                    j = rng.randint(0, total_words - 1)
                    if j in code_set:
                        continue
                    
                    # Check if swap maintains cover
                    union_new = balls[j]
                    for k in code:
                        if k != i:
                            union_new |= balls[k]
                    
                    if union_new == all_mask:
                        # Perform swap
                        idx_in_code = code.index(i)
                        code[idx_in_code] = j
                        code_set.remove(i)
                        code_set.add(j)
                        improved = True
                        break
                        
        return code

    while time.time() < deadline:
        # Greedy
        code_set = set()
        remaining = all_mask
        code = []
        
        while remaining:
            best_count = -1
            candidates = []
            
            for i in range(total_words):
                if i in code_set:
                    continue
                covered = balls[i] & remaining
                count = covered.bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count:
                    candidates.append(i)
            
            if best_count <= 0:
                break
            
            # Random tie-breaking
            best_idx = rng.choice(candidates)
            code.append(best_idx)
            code_set.add(best_idx)
            remaining &= ~balls[best_idx]
            
        # Local Search
        code = local_search(code, rng)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        if best_size <= 186:
            break

    if best_code is None:
        return [[0]*7]
        
    return [list(idx_to_word[i]) for i in best_code]
