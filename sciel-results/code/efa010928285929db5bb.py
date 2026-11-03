import time
import random
import math

def _standard_superperm(n):
    # Returns the standard superpermutation as a list of ints
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _extract_perms(word, n):
    # Extract unique permutations of 1..n in order of appearance
    perms = []
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms.append(window)
    return perms

def _build_word_from_order(perms, n, order):
    # perms: list of tuples
    # n: int
    # order: list of indices into perms
    # Returns the list of ints
    # We assume the 'base' word for n-1 is not needed, we just build the blocks
    # BUT, the standard construction relies on the 'base' word's structure for the initial part?
    # No, the recursive step replaces the entire word.
    # The new word is built solely from the blocks $P_i n P_i$.
    # Is this correct?
    # Let's check n=3.
    # Perms of 1,2: (1,2), (2,1).
    # Blocks: [1,2,3,1,2] and [2,1,3,2,1].
    # If order is [0, 1]:
    # New = [1,2,3,1,2]
    # Next block [2,1,3,2,1]. Overlap with [1,2,3,1,2] is 1 (2 matches).
    # New = [1,2,3,1,2,1,3,2,1].
    # This matches the standard construction.
    # So yes, we can build the word just from the blocks and the order.
    
    new = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(new), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new

def _optimize_order(perms, n, time_limit):
    k = len(perms)
    best_order = list(range(k))
    best_word = _build_word_from_order(perms, n, best_order)
    best_len = len(best_word)
    
    current_order = best_order[:]
    current_len = best_len
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Simple Hill Climbing / SA
    T = 10.0
    T_min = 0.1
    
    while time.time() < deadline - 1.0: # 1s safety margin
        # Generate neighbor
        i, j = random.sample(range(k), 2)
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_word = _build_word_from_order(perms, n, current_order)
        new_len = len(new_word)
        
        delta = new_len - current_len
        
        if delta < 0:
            current_len = new_len
            if new_len < best_len:
                best_len = new_len
                best_order = current_order[:]
                best_word = new_word
        else:
            # Accept with probability
            p = math.exp(-delta / T)
            if random.random() < p:
                current_len = new_len
        
        # Cool down
        T *= 0.999
        
    return best_word

def construct(n):
    if n <= 5:
        word = _standard_superperm(n)
        return "".join(map(str, word))
    
    # Strategy:
    # 1. Get n-1 superperm
    # 2. Extract perms
    # 3. Optimize order for n
    # For n=7, we might want to optimize n=6 first.
    # But time is tight. Let's try optimizing just the last step first.
    # If we have time, we can iterate.
    
    # For n=7, the base is n=6.
    # If we use the standard n=6, we might miss improvements in n=6.
    # Let's split time: 50% for n-1 optimization, 50% for n optimization.
    
    if n == 6:
        base_word = _standard_superperm(5)
        perms = _extract_perms(base_word, 5)
        # Optimize order for 6
        word = _optimize_order(perms, 6, 110)
        return "".join(map(str, word))
        
    elif n == 7:
        # First, try to get a good n=6
        # We have ~110s total.
        # Let's spend 40s on n=6 and 70s on n=7.
        
        # Step 1: Optimize n=6
        base_word_5 = _standard_superperm(5)
        perms_5 = _extract_perms(base_word_5, 5)
        # We need a function that returns the word, not just order
        # _optimize_order returns the word.
        word_6 = _optimize_order(perms_5, 6, 40)
        
        # Step 2: Extract perms from optimized n=6
        perms_6 = _extract_perms(word_6, 6)
        
        # Step 3: Optimize n=7
        word_7 = _optimize_order(perms_6, 7, 70)
        
        return "".join(map(str, word_7))
        
    else:
        # Fallback
        word = _standard_superperm(n)
        return "".join(map(str, word))
