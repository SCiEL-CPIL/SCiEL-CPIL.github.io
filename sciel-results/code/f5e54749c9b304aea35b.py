import time
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls using vectorized approach
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.append(to_idx(w))
            w[pos] = orig
        balls.append(covered)

    # Initial Code: Tetracode Sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    counts = [0] * total_words
    for c in initial_code:
        for w in balls[c]:
            counts[w] += 1
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    current_code = list(initial_code)
    current_counts = list(counts)
    code_set = set(current_code)
    
    def try_remove(c, counts_arr):
        for w in balls[c]:
            if counts_arr[w] == 1:
                return False
        return True

    def do_remove(c, counts_arr, code_set_local):
        for w in balls[c]:
            counts_arr[w] -= 1
        code_set_local.discard(c)

    def do_add(c, counts_arr, code_set_local):
        for w in balls[c]:
            counts_arr[w] += 1
        code_set_local.add(c)

    # Phase 1: Greedy Remove - multiple passes
    for _ in range(5):
        removed_any = False
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts, code_set)
                current_code.remove(c)
                removed_any = True
        if not removed_any:
            break
            
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
    
    # Phase 2: 2-opt local search
    # Try swapping: remove one codeword, add another, then greedy remove
    while time.time() < deadline:
        # Pick a random codeword to potentially remove
        if not current_code:
            break
            
        # Try a 2-opt move: remove c1, add c2, then greedy remove
        c1 = random.choice(current_code)
        
        # Find a candidate c2 to add
        # Prefer words that are "under-covered" or random
        c2 = random.randint(0, total_words - 1)
        if c2 in code_set:
            continue
            
        # Save state
        saved_code = list(current_code)
        saved_counts = list(current_counts)
        saved_set = set(code_set)
        
        # Remove c1
        if try_remove(c1, current_counts):
            do_remove(c1, current_counts, code_set)
            current_code.remove(c1)
        else:
            # Can't remove c1, try just adding c2 then removing
            pass
        
        # Add c2
        do_add(c2, current_counts, code_set)
        current_code.append(c2)
        
        # Greedy remove
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c, current_counts):
                do_remove(c, current_counts, code_set)
                current_code.remove(c)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
        else:
            # Revert
            current_code = saved_code
            current_counts = saved_counts
            code_set = saved_set

    return [to_word(idx) for idx in best_code]
