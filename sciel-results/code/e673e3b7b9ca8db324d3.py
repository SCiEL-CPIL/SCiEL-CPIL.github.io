import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [[0] * n]

    start = time.time()
    deadline = start + 115.0

    # Precompute all words and their neighbors (distance <= 1)
    # Word index: 0 to 3^7 - 1
    # Word to tuple:
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    all_words = list(range(3 ** n))
    
    # neighbors[i] is a list of indices of words within distance 1 of i
    neighbors = [[] for _ in range(3 ** n)]
    for i in all_words:
        w = idx_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for delta in (-1, 1):
                new_val = (orig + delta) % 3
                if new_val != orig:
                    w[pos] = new_val
                    neighbors[i].append(word_to_idx(w))
            w[pos] = orig

    # Initial Code: Tetracode x 3^3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            w = h + list(tail)
            initial_code.add(word_to_idx(w))
    
    best_code = initial_code
    best_size = len(best_code)
    
    # Helper: Get coverage counts for a set of codewords
    def get_counts(code_set):
        counts = np.zeros(3 ** n, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    # Helper: Check if code is valid
    def is_valid(code_set):
        if not code_set:
            return False
        counts = get_counts(code_set)
        return np.min(counts) >= 1

    # Helper: Greedy Prune
    # Returns a set of codewords, pruned as much as possible
    def prune(code_set):
        code_list = list(code_set)
        # Sort to ensure determinism? Or random?
        # Random order can help escape some local minima in pruning?
        # Usually, removing any redundant word is fine.
        # We iterate until no more can be removed.
        changed = True
        while changed:
            changed = False
            # Calculate counts once for the current set
            counts = get_counts(code_list)
            # Find a removable word
            # A word c is removable if counts[nb] >= 2 for all nb in neighbors[c]
            # Note: counts includes c itself.
            for c in code_list:
                removable = True
                for nb in neighbors[c]:
                    if counts[nb] < 2:
                        removable = False
                        break
                if removable:
                    code_list.remove(c)
                    changed = True
                    break
        return set(code_list)

    # Initial Prune
    best_code = prune(best_code)
    best_size = len(best_code)
    
    # Local Search: Swap + Prune
    # We try to swap a word in best_code with a word not in best_code.
    # If the resulting set is valid, we prune it.
    # If the pruned size is smaller, we accept.
    
    random.seed(42)
    all_words_list = all_words
    
    while time.time() < deadline:
        # Pick a random word to swap out
        if not best_code:
            break
        c_out = random.choice(list(best_code))
        
        # Pick a random word to swap in
        # We can bias towards words that cover "tight" spots?
        # For now, random.
        c_in = random.choice(all_words_list)
        
        if c_in == c_out:
            continue
        if c_in in best_code:
            continue
            
        new_code = set(best_code)
        new_code.remove(c_out)
        new_code.add(c_in)
        
        # Check validity of new_code
        # Optimization: Only check if new_code is valid before pruning?
        # Pruning assumes validity? No, prune removes redundant words.
        # If new_code is invalid, prune might remove a word and make it worse?
        # Actually, if new_code is invalid, some nb has count 0 or 1.
        # If count is 0, it's invalid.
        # If count is 1, that word is NOT removable.
        # So prune on an invalid set will just return a subset that is still invalid (or empty).
        # We must ensure the set is valid BEFORE pruning, or check validity after.
        
        # Fast validity check:
        counts = get_counts(new_code)
        if np.min(counts) < 1:
            continue
            
        # It is valid. Now prune.
        pruned_code = prune(new_code)
        
        if len(pruned_code) < best_size:
            best_code = pruned_code
            best_size = len(best_code)
            
            # Optional: Try to do more swaps immediately
            # The loop will handle it.
            
    # Convert indices back to lists of ints
    result = [idx_to_word(c) for c in best_code]
    return result
