"""Shortest superpermutation solver using classic construction + local search shortening."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """Construct shortest superpermutation over symbols 1..n."""
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s kill
    
    if n == 1:
        return "1"
    
    if n <= 5:
        return _classic_construct(n)
    
    # Start with classic construction
    word_str = _classic_construct(n)
    word = [int(c) for c in word_str]
    
    # Local search: try to remove redundant symbols
    word = _local_search_shorten(word, n, start_time, deadline)
    
    return "".join(map(str, word))


def _classic_construct(n):
    """Classic recursive construction of superpermutation."""
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _local_search_shorten(word, n, start_time, deadline):
    """Try to shorten the superpermutation by removing redundant symbols."""
    time_limit = deadline - (time.time() - start_time)
    if time_limit < 5:
        return word
    
    # Build set of all required permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    def get_windows(w):
        """Get all length-n windows as tuples."""
        return [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
    
    def is_valid(w):
        """Check if w contains all permutations as consecutive windows."""
        windows = set()
        for i in range(len(w) - n + 1):
            windows.add(tuple(w[i:i+n]))
        return all_perms.issubset(windows)
    
    # Verify initial validity
    if not is_valid(word):
        return word
    
    best_word = list(word)
    best_len = len(word)
    
    # Try removing each symbol and check validity
    # Process in random order for diversity
    indices = list(range(len(word)))
    
    # Multiple passes
    max_passes = 10
    for _pass in range(max_passes):
        if time.time() - start_time > deadline - 5:
            break
        
        random.shuffle(indices)
        improved = False
        
        for idx in indices:
            if time.time() - start_time > deadline - 5:
                break
            
            if idx >= len(word):
                continue
            
            # Try removing word[idx]
            test_word = word[:idx] + word[idx+1:]
            if len(test_word) < n:
                continue
            
            if is_valid(test_word):
                word = test_word
                improved = True
                # After removal, indices shift; restart this pass
                break
        
        if not improved:
            break
        
        if len(word) < best_len:
            best_word = list(word)
            best_len = len(word)
    
    # Second phase: try removing pairs of adjacent symbols
    if time.time() - start_time < deadline - 10 and len(word) > best_len:
        word = list(best_word)
        best_len = len(word)
    
    if time.time() - start_time < deadline - 10:
        for _pass in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            for idx in range(len(word) - 1):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Try removing word[idx] and word[idx+1]
                test_word = word[:idx] + word[idx+2:]
                if len(test_word) < n:
                    continue
                
                if is_valid(test_word):
                    word = test_word
                    improved = True
                    break
            
            if not improved:
                break
            
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
    
    return best_word
# EVOLVE-BLOCK-END
